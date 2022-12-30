/*
 * rvmofx.cpp
 *
 * vim: ts=8 sw=8
 *
 * Main entry point for plugin
 *
 * Copyright (c) 2022-2023 Sylvain Munaut <tnt@246tNt.com>
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <cstring>
#include <iostream>
#include <stdexcept>

#include <torch/script.h>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxPixels.h"

#if defined __APPLE__ || defined linux || defined __FreeBSD__
#  define EXPORT OfxExport __attribute__((visibility("default")))
#else
#  define EXPORT OfxExport
#endif


/* ------------------------------------------------------------------------- */
/* Globals                                                                   */
/* ------------------------------------------------------------------------- */

/* pointers to various bits of the host */
static OfxHost *		gHost;
static OfxImageEffectSuiteV1 *	gEffectHost;
static OfxPropertySuiteV1 *	gPropHost;
static OfxParameterSuiteV1 *	gParamHost;

static char *gBundlePath;



/* ------------------------------------------------------------------------- */
/* Private Data                                                              */
/* ------------------------------------------------------------------------- */

enum deviceParamValue {
	DEVICE_CPU = 0,
	DEVICE_CUDA = 1,
};

enum modelParamValue {
	MODEL_MOBILENETV3 = 0,
	MODEL_RESNET50 = 1,
	MODEL_CUSTOM = 2,
};

enum modelPrecisionParamValue {
	MODEL_PRECISION_FLOAT16 = 0,
	MODEL_PRECISION_FLOAT32 = 1,
};

enum outputTypeParamValue {
	OUTPUT_RGBA  = 0,
	OUTPUT_ALPHA = 1,
};

enum colorSourceParamValue {
	COLOR_SRC_INPUT = 0,
	COLOR_SRC_MODEL = 1,
};


struct InstanceData {
	/* Clips Handles */
	OfxImageClipHandle outputClip;
	OfxImageClipHandle inputClip;
	OfxImageClipHandle garbageMatteClip;
	OfxImageClipHandle solidMatteClip;

	/* Params Handles */
	OfxParamHandle deviceParam;
	OfxParamHandle modelParam;
	OfxParamHandle modelFileParam;
	OfxParamHandle modelPrecisionParam;
	OfxParamHandle downsampleRatioParam;
	OfxParamHandle outputTypeParam;
	OfxParamHandle colorSourceParam;
	OfxParamHandle postmultiplyAlphaParam;

	/* Cached values */
	bool   hasGarbageMatte;
	bool   hasSolidMatte;

	double downsampleRatio;
	enum outputTypeParamValue outputType;
	enum colorSourceParamValue colorSource;
	bool postmultiplyAlpha;

	/* TorchScript */
	struct _torch {
		bool ready;

		torch::Device device;
		torch::Dtype type;

		torch::jit::script::Module model;

		OfxTime rn_time;
		torch::Tensor rn[4];

		_torch() : device(torch::kCPU), rn_time(nan("")) {}; /* workaround for torch::Device requiring init ... */
	} torch;
};

static InstanceData *
getInstanceData(OfxImageEffectHandle effect)
{
	InstanceData *priv = NULL;
	OfxPropertySetHandle effectProps;

	gEffectHost->getPropertySet(effect, &effectProps);
	gPropHost->propGetPointer(effectProps,
		kOfxPropInstanceData, 0,
		(void **) &priv);

	return priv;
}



/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */

static inline void
setParamEnabledness(
	OfxImageEffectHandle effect,
	const char *paramName, bool enabledState)
{
	/* Fetch the parameter set for this effect */
	OfxParamSetHandle paramSet;
	gEffectHost->getParamSet(effect, &paramSet);

	/* Fetch the parameter property handle */
	OfxParamHandle param; OfxPropertySetHandle paramProps;
	gParamHost->paramGetHandle(paramSet, paramName, &param, &paramProps);

	/* And set its enabledness */
	gPropHost->propSetInt(paramProps, kOfxParamPropEnabled, 0, int(enabledState));
}

static void
updateParamsValidity(OfxImageEffectHandle effect)
{
	InstanceData *priv = getInstanceData(effect);

	/* Compute device affects precision */
	enum deviceParamValue dev;
	gParamHost->paramGetValue(priv->deviceParam, &dev);

	switch (dev) {
	case DEVICE_CPU:
		gParamHost->paramSetValue(priv->modelPrecisionParam, int(MODEL_PRECISION_FLOAT32));
		setParamEnabledness(effect, "modelPrecision", false);
		break;
	case DEVICE_CUDA:
		setParamEnabledness(effect, "modelPrecision", true);
		break;
	}

	/* Model -> ModelFile */
	int model;
	gParamHost->paramGetValue(priv->modelParam, &model);
	setParamEnabledness(effect, "modelFile", (model == MODEL_CUSTOM));

	/* OutputType -> ColorSource / PostMultiply */
	enum outputTypeParamValue output_type;
	gParamHost->paramGetValue(priv->outputTypeParam, &output_type);
	setParamEnabledness(effect, "colorSource", (output_type == OUTPUT_RGBA));
	setParamEnabledness(effect, "postmultiplyAlpha", (output_type == OUTPUT_RGBA));
}

static const char *
getModelFilename(OfxImageEffectHandle effect)
{
	static char path[PATH_MAX];

	InstanceData *priv = getInstanceData(effect);

	/* Model */
	int model;
	enum modelPrecisionParamValue model_precision;
	char *model_file;

	gParamHost->paramGetValue(priv->modelParam, &model);
	gParamHost->paramGetValue(priv->modelPrecisionParam, &model_precision);
	gParamHost->paramGetValue(priv->modelFileParam, &model_file);

	/* Build path */
	switch (model) {
	case MODEL_MOBILENETV3:
		snprintf(path, PATH_MAX, "%s/Contents/Resources/rvm_mobilenetv3_fp%d.torchscript",
			gBundlePath,
			(model_precision == MODEL_PRECISION_FLOAT16) ? 16 : 32
		);
		break;

	case MODEL_RESNET50:
		snprintf(path, PATH_MAX, "%s/Contents/Resources/rvm_resnet50_fp%d.torchscript",
			gBundlePath,
			(model_precision == MODEL_PRECISION_FLOAT16) ? 16 : 32
		);
		break;

	case MODEL_CUSTOM:
		if (!model_file || !model_file[0])
			return NULL;

		snprintf(path, PATH_MAX, "%s", model_file);
		break;
	}

	return path;
}

static void
modelClearHistory(OfxImageEffectHandle effect)
{
	InstanceData *priv = getInstanceData(effect);

	if (std::isnan(priv->torch.rn_time))
		return;

	priv->torch.rn_time = nan("");

	for (int i=0; i<4; i++)
		priv->torch.rn[i] = torch::Tensor();
}

static OfxStatus
modelSetup(OfxImageEffectHandle effect)
{
	InstanceData *priv = getInstanceData(effect);

	if (priv->torch.ready)
		return kOfxStatOK;

	/* Target device and type from config */
	enum deviceParamValue dev;
	enum modelPrecisionParamValue precision;

	gParamHost->paramGetValue(priv->deviceParam, &dev);
	gParamHost->paramGetValue(priv->modelPrecisionParam, &precision);

	switch (dev) {
	case DEVICE_CPU:
		priv->torch.device = torch::Device(torch::kCPU);
		break;
	case DEVICE_CUDA:
		priv->torch.device = torch::Device(torch::kCUDA);
		break;
	}

	switch (precision) {
	case MODEL_PRECISION_FLOAT16:
    		priv->torch.type = torch::kFloat16;
		break;
	case MODEL_PRECISION_FLOAT32:
    		priv->torch.type = torch::kFloat32;
		break;
	}

	/* Load model */
	const char *model_file = getModelFilename(effect);

	if (!model_file)
		return kOfxStatFailed;

	try {
		priv->torch.model = torch::jit::load(model_file);
		priv->torch.model.to(priv->torch.device);
		torch::jit::freeze(priv->torch.model);
		torch::jit::getProfilingMode() = false;
	} catch (const std::exception& e) {
		std::cerr << "[!] OFX Plugin error: Exception caught while loading model: " << e.what() << std::endl;
		return kOfxStatFailed;
	}

	/* Reset recursive state */
	modelClearHistory(effect);

	/* We're ready */
	priv->torch.ready = true;

	return kOfxStatOK;
}


/* ------------------------------------------------------------------------- */
/* API Handlers                                                              */
/* ------------------------------------------------------------------------- */

static OfxStatus
effectLoad(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	/* Fetch the host suites out of the global host pointer */
	if(!gHost)
		return kOfxStatErrMissingHostFeature;

	gEffectHost     = (OfxImageEffectSuiteV1 *) gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1);
	gPropHost       = (OfxPropertySuiteV1 *)    gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1);
	gParamHost	= (OfxParameterSuiteV1 *)   gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1);
	if(!gEffectHost || !gPropHost || !gParamHost)
		return kOfxStatErrMissingHostFeature;

	return kOfxStatOK;
}


static OfxStatus
effectUnload(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	/* Reset */
	gEffectHost = NULL;
	gPropHost   = NULL;
	gParamHost  = NULL;

	free(gBundlePath);
	gBundlePath = NULL;

	return kOfxStatOK;
}


static OfxStatus
effectCreateInstance(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	/* Get a pointer to the effect properties */
	OfxPropertySetHandle effectProps;
	gEffectHost->getPropertySet(effect, &effectProps);

	/* Get a pointer to the effect's parameter set */
	OfxParamSetHandle paramSet;
	gEffectHost->getParamSet(effect, &paramSet);

	/* Create private instance data holder */
	InstanceData *priv = new InstanceData();

	/* Cache away clip handles */
	gEffectHost->clipGetHandle(effect, kOfxImageEffectOutputClipName, &priv->outputClip, 0);
	gEffectHost->clipGetHandle(effect, "Input",                       &priv->inputClip, 0);
	gEffectHost->clipGetHandle(effect, "GarbageMatte",                &priv->garbageMatteClip, 0);
	gEffectHost->clipGetHandle(effect, "SolidMatte",                  &priv->solidMatteClip, 0);

	/* Cache away param handles */
	gParamHost->paramGetHandle(paramSet, "device",             &priv->deviceParam, 0);
	gParamHost->paramGetHandle(paramSet, "model",              &priv->modelParam, 0);
	gParamHost->paramGetHandle(paramSet, "modelFile",          &priv->modelFileParam, 0);
	gParamHost->paramGetHandle(paramSet, "modelPrecision",     &priv->modelPrecisionParam, 0);
	gParamHost->paramGetHandle(paramSet, "downsampleRatio",    &priv->downsampleRatioParam, 0);
	gParamHost->paramGetHandle(paramSet, "outputType",         &priv->outputTypeParam, 0);
	gParamHost->paramGetHandle(paramSet, "colorSource",        &priv->colorSourceParam, 0);
	gParamHost->paramGetHandle(paramSet, "postmultiplyAlpha",  &priv->postmultiplyAlphaParam, 0);

	/* Set private instance data */
	gPropHost->propSetPointer(effectProps, kOfxPropInstanceData, 0, (void *) priv);

	/* Update wiht loaded params values */
	updateParamsValidity(effect);

	return kOfxStatOK;
}


static OfxStatus
effectDestroyInstance(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	InstanceData *priv = getInstanceData(effect);

	if (priv)
		delete priv;

	return kOfxStatOK;
}


static OfxStatus
effectInstanceChanged(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	InstanceData *priv = getInstanceData(effect);

	/* See why it changed */
	char *changeReason;
	gPropHost->propGetString(inArgs, kOfxPropChangeReason, 0, &changeReason);

	/* We are only interested in user edits */
	if (strcmp(changeReason, kOfxChangeUserEdited) != 0)
		return kOfxStatReplyDefault;

	/* Fetch the type & name of the object that changed */
	char *typeChanged;
	gPropHost->propGetString(inArgs, kOfxPropType, 0, &typeChanged);

	bool isClip  = strcmp(typeChanged, kOfxTypeClip) == 0;
	bool isParam = strcmp(typeChanged, kOfxTypeParameter) == 0;

	char *objChanged;
	gPropHost->propGetString(inArgs, kOfxPropName, 0, &objChanged);

	/* Some changes invalidate things */
	if (isParam && (
	    !strcmp(objChanged, "device") ||
	    !strcmp(objChanged, "model") ||
	    !strcmp(objChanged, "modelPrecision") ||
	    !strcmp(objChanged, "modelFile"))) {
	    	priv->torch.ready = false;	/* Reload model */
		return kOfxStatOK;
	}

	if (isParam && (
	    !strcmp(objChanged, "downsampleRatio"))) {
		modelClearHistory(effect);	/* Recursive history invalidate */
		return kOfxStatOK;
	}

	/* Change in clips */
	if (isClip) {
		OfxImageClipHandle clip;
		OfxPropertySetHandle props;
		int connected;

		gEffectHost->clipGetHandle(effect, objChanged, &clip, &props);
		gPropHost->propGetInt(props,  kOfxImageClipPropConnected, 0, &connected);

		/* Input -> Invalidate recursive history */
		if (strcmp(objChanged, "Input")) {
			modelClearHistory(effect);
			return kOfxStatOK;
		}

		/* GarbageMatte / SolidMatte -> Check if connected */
		if (strcmp(objChanged, "GarbageMatte")) {
			priv->hasGarbageMatte = connected;
			return kOfxStatOK;
		}

		if (isClip && !strcmp(objChanged, "SolidMatte")) {
			priv->hasSolidMatte = connected;
			return kOfxStatOK;
		}
	}

	/* Don't trap any others */
	return kOfxStatReplyDefault;
}


static OfxStatus
effectEndInstanceChanged(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	InstanceData *priv = getInstanceData(effect);

	/* If it's a user edit : Update enabled params */
	char *changeReason;
	gPropHost->propGetString(inArgs, kOfxPropChangeReason, 0, &changeReason);
	if (!strcmp(changeReason, kOfxChangeUserEdited))
		updateParamsValidity(effect);

	/* Update cached param values int all cases */
	gParamHost->paramGetValue(priv->downsampleRatioParam, &priv->downsampleRatio);
	gParamHost->paramGetValue(priv->outputTypeParam, &priv->outputType);
	gParamHost->paramGetValue(priv->colorSourceParam, &priv->colorSource);
	gParamHost->paramGetValue(priv->postmultiplyAlphaParam, &priv->postmultiplyAlpha);

	/* Done */
	return kOfxStatOK;
}


static OfxStatus
effectDescribe(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	/* Get the property handle for the plugin */
	OfxPropertySetHandle effectProps;
	gEffectHost->getPropertySet(effect, &effectProps);

	/* Identity / Classification */
	gPropHost->propSetString(effectProps, kOfxPropLabel, 0, "OFX Robust Video Matting");
	gPropHost->propSetString(effectProps, kOfxImageEffectPluginPropGrouping, 0, "OpenFX");

	/* Applicable contexts */
	gPropHost->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextGeneral);

	/* We expect float images as inputs */
	gPropHost->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);

	/* Don't allow tiling, we need the full images at once */
	gPropHost->propSetInt(effectProps, kOfxImageEffectPropSupportsTiles, 0, 0);

	/* We need to render things in sequence */
	gPropHost->propSetInt(effectProps, kOfxImageEffectInstancePropSequentialRender, 0, 1);

	/* Parameters that affect clip preferences */
	gPropHost->propSetString(effectProps, kOfxImageEffectPropClipPreferencesSlaveParam, 0, "outputType");
	gPropHost->propSetString(effectProps, kOfxImageEffectPropClipPreferencesSlaveParam, 0, "postmultiplyAlpha");

	return kOfxStatOK;
}


static OfxStatus
effectDescribeInContext(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	OfxPropertySetHandle props;

	/* Check it's kOfxImageEffectContextGeneral */
	char *context;
	gPropHost->propGetString(inArgs, kOfxImageEffectPropContext, 0, &context);
	if (strcmp(context, kOfxImageEffectContextGeneral))
		return kOfxStatErrFatal;

	/* Get the path to bundle */
	/* (apparently you have to get it from here ...) */
	if (!gBundlePath) {
		OfxPropertySetHandle effectProps;
		gEffectHost->getPropertySet(effect, &effectProps);
		gPropHost->propGetString(effectProps, kOfxPluginPropFilePath, 0, &gBundlePath);
		gBundlePath = strdup(gBundlePath);
	}

	/* Clips (names copied from DeltaKeyer) */
		/* Output clip */
	gEffectHost->clipDefine(effect, kOfxImageEffectOutputClipName, &props);

	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);

		/* Input clip */
	gEffectHost->clipDefine(effect, "Input", &props);

	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGB);
	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGBA);

		/* Garbage Matte */
	gEffectHost->clipDefine(effect, "GarbageMatte", &props);

	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentNone);
	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
	gPropHost->propSetInt(props, kOfxImageClipPropOptional, 0, 1);

		/* Solid Matte */
	gEffectHost->clipDefine(effect, "SolidMatte", &props);

	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentNone);
	gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
	gPropHost->propSetInt(props, kOfxImageClipPropOptional, 0, 1);

	/* Parameters */
	OfxParamSetHandle paramSet;
	gEffectHost->getParamSet(effect, &paramSet);

		/* Compute Device */
	gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, "device", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Compute Device");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "What device backend to use to run model");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, DEVICE_CPU,  "CPU");
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, DEVICE_CUDA, "CUDA");

		/* Model */
	gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, "model", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Model");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "What model to load for backbone (either default/prebuilt, or custome one");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, MODEL_MOBILENETV3, "mobilenetv3");
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, MODEL_RESNET50,    "resnet50");
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, MODEL_CUSTOM,      "custom");

		/* Model File (custom) */
	gParamHost->paramDefine(paramSet, kOfxParamTypeString, "modelFile", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Model File");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "Path to model filename");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);
	gPropHost->propSetString(props, kOfxParamPropStringMode, 0, kOfxParamStringIsFilePath);
	gPropHost->propSetInt   (props, kOfxParamPropEnabled, 0, 0);

		/* Model Precision */
	gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, "modelPrecision", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Model Precision");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "Precision to use (for custom models, must match file !)");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, MODEL_PRECISION_FLOAT16, "float16");
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, MODEL_PRECISION_FLOAT32, "float32");
	gPropHost->propSetInt   (props, kOfxParamPropDefault, 0, MODEL_PRECISION_FLOAT32);
	gPropHost->propSetInt   (props, kOfxParamPropEnabled, 0, 0);

		/* Downsample ratio */
	gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, "downsampleRatio", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Downsample ratio");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "Image downsampling ratio. Set to 0.0 for model auto-select");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);
	gPropHost->propSetString(props, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeScale);
	gPropHost->propSetDouble(props, kOfxParamPropMin, 0, 0.0);
	gPropHost->propSetDouble(props, kOfxParamPropMax, 0, 1.0);
	gPropHost->propSetDouble(props, kOfxParamPropDefault, 0, 0.0);

		/* Output type */
	gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, "outputType", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Output type");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "Selects between full RGBA output or mask-only output");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, OUTPUT_RGBA,  "RGBA");
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, OUTPUT_ALPHA, "Alpha");

		/* Color source */
	gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, "colorSource", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Output Color Source");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "Selects whether to use the input RGB value or the model predicted foreground for the output color components");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, COLOR_SRC_INPUT,  "Input Clip");
	gPropHost->propSetString(props, kOfxParamPropChoiceOption, COLOR_SRC_MODEL, "Model Prediction");
	gPropHost->propSetInt   (props, kOfxParamPropDefault, 0, COLOR_SRC_MODEL);

		/* Post-multiply alpha */
	gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, "postmultiplyAlpha", &props);
	gPropHost->propSetString(props, kOfxPropLabel, 0, "Output Postmultiply Alpha");
	gPropHost->propSetString(props, kOfxParamPropHint, 0, "Enable/Disable multiplying RGB with Alpha on the output");
	gPropHost->propSetInt   (props, kOfxParamPropAnimates, 0, 0);

	return kOfxStatOK;
}


static OfxStatus
effectGetClipPreferences(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	InstanceData *priv = getInstanceData(effect);
	OfxStatus status = kOfxStatOK;

	/* Input clip */
		/* FIXME should we check that kOfxImageClipPropUnmappedComponents is RGB / RGBA ? */
	gPropHost->propSetString(outArgs, "OfxImageClipPropDepth_Input", 0, kOfxBitDepthFloat);

	/* GarbageMatte / SolidMatte clips */
	if (priv->hasGarbageMatte) {
		gPropHost->propSetString(outArgs, "OfxImageClipPropComponents_GarbageMatte", 0, kOfxImageComponentAlpha);
		gPropHost->propSetString(outArgs, "OfxImageClipPropDepth_GarbageMatte", 0, kOfxBitDepthFloat);
	}

	if (priv->hasSolidMatte) {
		gPropHost->propSetString(outArgs, "OfxImageClipPropComponents_SolidMatte", 0, kOfxImageComponentAlpha);
		gPropHost->propSetString(outArgs, "OfxImageClipPropDepth_SolidMatte", 0, kOfxBitDepthFloat);
	}

	/* Output clip */
		/* Get config */
	enum outputTypeParamValue out_type;
	bool postmultiply_alpha;

	gParamHost->paramGetValue(priv->outputTypeParam, &out_type);
	gParamHost->paramGetValue(priv->postmultiplyAlphaParam, &postmultiply_alpha);

		/* Components / Depth */
	gPropHost->propSetString(outArgs, "OfxImageClipPropComponents_Output", 0,
		(out_type == OUTPUT_ALPHA) ? kOfxImageComponentAlpha : kOfxImageComponentRGBA);
	gPropHost->propSetString(outArgs, "OfxImageClipPropDepth_Output", 0, kOfxBitDepthFloat);

		/* Alpha Premultiplied ? */
	gPropHost->propSetString(outArgs, kOfxImageEffectPropPreMultiplication, 0,
		postmultiply_alpha ? kOfxImagePreMultiplied : kOfxImageUnPreMultiplied);

	return status;
}


static OfxStatus
effectBeginSequenceRender(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	InstanceData *priv = getInstanceData(effect);
	OfxStatus status = kOfxStatOK;

	(void)priv;

#if 0
	int is_interactive, sequential_status, interactive_status;

	gPropHost->propGetInt(inArgs, kOfxPropIsInteractive, 0, &is_interactive);
	gPropHost->propGetInt(inArgs, kOfxImageEffectPropSequentialRenderStatus,  0, &sequential_status);
	gPropHost->propGetInt(inArgs, kOfxImageEffectPropInteractiveRenderStatus, 0, &interactive_status);

	printf("Begin: is_interactive=%d, sequential_status=%d, interactive_status=%d\n",
		is_interactive,
		sequential_status,
		interactive_status
	);
#endif

	return status;
}

static OfxStatus
effectEndSequenceRender(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	InstanceData *priv = getInstanceData(effect);
	OfxStatus status = kOfxStatOK;

	(void)priv;

	return status;
}


class NoImageEx {};

struct ImageInfo {
	OfxPropertySetHandle h;
	OfxRectI rect;
	int rowBytes;
	void *ptr;
	char *pixelDepth;
	char *components;
};

static OfxStatus
fillImageInfos(
	struct ImageInfo &img,
	OfxImageEffectHandle effect,
	OfxImageClipHandle clip,
	OfxTime time)
{
	OfxStatus rv;

	rv = gEffectHost->clipGetImage(clip, time, NULL, &img.h);
	if (rv != kOfxStatOK)
		return rv;

	gPropHost->propGetIntN   (img.h, kOfxImagePropBounds,   4, &img.rect.x1);
	gPropHost->propGetInt    (img.h, kOfxImagePropRowBytes, 0, &img.rowBytes);
	gPropHost->propGetPointer(img.h, kOfxImagePropData,     0, &img.ptr);
	gPropHost->propGetString (img.h, kOfxImageEffectPropPixelDepth, 0, &img.pixelDepth);
	gPropHost->propGetString (img.h, kOfxImageEffectPropComponents, 0, &img.components);

	return kOfxStatOK;
}

static torch::Tensor
imageToTensor(struct ImageInfo &img, torch::Device td, torch::Dtype tt)
{
	int w = img.rect.x2 - img.rect.x1;
	int h = img.rect.y2 - img.rect.y1;
	int nc, vs;
	uint8_t *p;
	torch::Dtype dt;
	float sf = 1.0f;

	if (!strcmp(img.components, kOfxImageComponentRGBA)) {
		nc = 4;
	} else if (!strcmp(img.components, kOfxImageComponentRGB )) {
		nc = 3;
	} else if (!strcmp(img.components, kOfxImageComponentAlpha)) {
		nc = 1;
	} else {
		return torch::Tensor();
	}

	if (!strcmp(img.pixelDepth, kOfxBitDepthByte)) {
		vs = 1;
		dt = torch::kByte;
		sf = 1.0f / 255.0f;
	} else if (!strcmp(img.pixelDepth, kOfxBitDepthShort)) {
		vs = 2;
		dt = torch::kShort;
		sf = 1.0f / 32768.0f;
	} else if (!strcmp(img.pixelDepth, kOfxBitDepthHalf)) {
		vs = 2;
		dt = torch::kFloat16;
	} else if (!strcmp(img.pixelDepth, kOfxBitDepthFloat)) {
		vs = 4;
		dt = torch::kFloat32;
	} else {
		return torch::Tensor();
	}

	p = ((uint8_t*)img.ptr) + img.rowBytes * img.rect.y1 + vs * img.rect.x1;

	torch::Tensor rv = torch::from_blob(
		p,
		{ h, w, nc },
		{ img.rowBytes / vs, nc, 1 },
		dt
	);

	rv = rv.to(td, tt);
	if (sf != 1.0f)
		rv *= sf;
	rv = rv.permute({2, 0, 1});
	rv = rv.unsqueeze(0);

	return rv;
}

static void
tensorToImage(struct ImageInfo &img, torch::Tensor t)
{
	int w = img.rect.x2 - img.rect.x1;
	int h = img.rect.y2 - img.rect.y1;
	int nc, vs;
	uint8_t *p_dst, *p_src;
	torch::ScalarType dt;
	torch::Device dev_cpu = torch::Device("cpu");
	float sf = 1.0f;

	if (!strcmp(img.components, kOfxImageComponentRGBA)) {
		nc = 4;
	} else if (!strcmp(img.components, kOfxImageComponentRGB )) {
		nc = 3;
	} else if (!strcmp(img.components, kOfxImageComponentAlpha)) {
		nc = 1;
	} else {
		return;
	}

	if (!strcmp(img.pixelDepth, kOfxBitDepthByte)) {
		vs = 1;
		dt = torch::kByte;
		sf = 255.0f;
	} else if (!strcmp(img.pixelDepth, kOfxBitDepthShort)) {
		vs = 2;
		dt = torch::kShort;
		sf = 32768.0f;
	} else if (!strcmp(img.pixelDepth, kOfxBitDepthHalf)) {
		vs = 2;
		dt = torch::kFloat16;
	} else if (!strcmp(img.pixelDepth, kOfxBitDepthFloat)) {
		vs = 4;
		dt = torch::kFloat32;
	} else {
		return;
	}

	t = t.squeeze(0);
	t = t.permute({1, 2, 0});
	if (sf != 1.0f)
		t *= sf;
	t = t.to(dev_cpu, dt);
	t = t.contiguous();

	p_src = (uint8_t*) t.data_ptr<float>();
	p_dst = ((uint8_t*)img.ptr) + img.rowBytes * img.rect.y1 + vs * img.rect.x1;

	for (int y=0; y<h; y++) {
		memcpy(p_dst, p_src, w*nc*vs);
		p_src += vs * t.strides()[0];
		p_dst += img.rowBytes;
	}
}


static OfxStatus
effectRender(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	InstanceData *priv = getInstanceData(effect);

	OfxTime time;
	OfxRectI renderWindow;
	OfxStatus status = kOfxStatOK;

	/* Target time and window */
	gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);
	gPropHost->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &renderWindow.x1);

	/* Prepare the model */
	status = modelSetup(effect);
	if (status != kOfxStatOK)
		return status;

	/* */
	ImageInfo outputImg;
	ImageInfo inputImg;

	try {
		torch::NoGradGuard no_grad_guard;

		/* Get images */
		if (fillImageInfos(outputImg, effect, priv->outputClip, time) != kOfxStatOK)
			throw NoImageEx();
		if (fillImageInfos(inputImg,  effect, priv->inputClip,  time) != kOfxStatOK)
			throw NoImageEx();

#if 0
		printf("R: %d %d %d %d\n", renderWindow.x1, renderWindow.x2, renderWindow.y1, renderWindow.y2);
		printf("O: %d %d %d %d %s %s\n", outputImg.rect.x1, outputImg.rect.x2, outputImg.rect.y1, outputImg.rect.y2, outputImg.pixelDepth, outputImg.components);
		printf("I: %d %d %d %d %s %s\n", inputImg.rect.x1, inputImg.rect.x2, inputImg.rect.y1, inputImg.rect.y2, inputImg.pixelDepth, inputImg.components);
#endif

		/* OFX Image -> Input tensor */
		auto inputTensor = imageToTensor(inputImg, priv->torch.device, priv->torch.type);
		if (!inputTensor.defined())
			throw NoImageEx();

		switch (inputTensor.sizes()[1])
		{
		case 3: /* RGB already, nothing to do */
			break;

		case 4: /* RGBA, drop alpha */
			inputTensor = inputTensor.narrow(1,0,3);
			break;

		default: /* Other ? Can't deal with that */
			throw NoImageEx();
		}

		/* Run the model */
		torch::jit::Kwargs kwargs;
		c10::List<torch::Tensor> outputs;

		if (priv->downsampleRatio != 0.0) {
			kwargs.insert({"downsample_ratio", priv->downsampleRatio});
		}

		if (priv->torch.rn[0].defined() && (
			(time == (priv->torch.rn_time + 1.0)) ||
			(time ==  priv->torch.rn_time)
		)) {
			/* We have usable recursive states */
			outputs = priv->torch.model.forward({
				inputTensor,
				priv->torch.rn[0],
				priv->torch.rn[1],
				priv->torch.rn[2],
				priv->torch.rn[3]
			}, kwargs).toTensorList();
		}
		else
		{
			/* First of a sequence of run */
			outputs = priv->torch.model.forward({
				inputTensor
			}, kwargs).toTensorList();
		}

		/* Recursive states for next run */
		priv->torch.rn_time = time;

		priv->torch.rn[0] = outputs.get(2);
		priv->torch.rn[1] = outputs.get(3);
		priv->torch.rn[2] = outputs.get(4);
		priv->torch.rn[3] = outputs.get(5);

		/* Tensor outputs */
		auto fgr = outputs.get(0);
		auto pha = outputs.get(1);

		/* Post process of output tensor depending on options */
		torch::Tensor outputTensor;

		switch (priv->outputType)
		{
		case OUTPUT_RGBA:
			/* Select foreground */
			if (priv->colorSource == COLOR_SRC_INPUT)
				fgr = inputTensor;

			/* Post Multiply ? */
			if (priv->postmultiplyAlpha) {
				fgr *= pha.repeat({1, 3, 1, 1});
			}

			/* Combine */
			outputTensor = torch::cat({fgr, pha}, 1);

			break;

		case OUTPUT_ALPHA:
			/* Repeat the alpha as needed by output components */
			if (!strcmp(outputImg.components, kOfxImageComponentRGBA)) {
				outputTensor = pha.repeat({1, 4, 1, 1});
			} else if (!strcmp(outputImg.components, kOfxImageComponentRGB )) {
				outputTensor = pha.repeat({1, 3, 1, 1});
			} else if (!strcmp(outputImg.components, kOfxImageComponentAlpha)) {
				outputTensor = pha;
			} else {
				throw NoImageEx();
			}
			break;

		default:
			throw NoImageEx();
		}

		/* Output tensor -> OFX Image */
		tensorToImage(outputImg, outputTensor);

#if 0
		std::cout << "tensor dtype   = " << inputTensor.dtype()   << std::endl;
		std::cout << "tensor size    = " << inputTensor.sizes()   << std::endl;
		std::cout << "tensor strides = " << inputTensor.strides() << std::endl;
#endif

	} catch(NoImageEx &) {
		/* Missing a required clip, so abort */
		if(!gEffectHost->abort(effect)) {
			status = kOfxStatFailed;
		}
	}

	/* Cleanup */
	if (outputImg.h)
		gEffectHost->clipReleaseImage(outputImg.h);
	if (inputImg.h)
		gEffectHost->clipReleaseImage(inputImg.h);

	return status;
}


typedef OfxStatus (*handler_func_t)(
	OfxImageEffectHandle effect,
	OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs
);

static const struct {
	const char *action;
	handler_func_t handler;
} _apiHandlers[] = {
	{ kOfxActionLoad, effectLoad},
	{ kOfxActionUnload, effectUnload},
	{ kOfxActionCreateInstance, effectCreateInstance },
	{ kOfxActionDestroyInstance, effectDestroyInstance },
	{ kOfxActionInstanceChanged, effectInstanceChanged },
	{ kOfxActionEndInstanceChanged, effectEndInstanceChanged },
	{ kOfxActionDescribe, effectDescribe},
	{ kOfxImageEffectActionDescribeInContext, effectDescribeInContext},
	{ kOfxImageEffectActionGetClipPreferences, effectGetClipPreferences },
	{ kOfxImageEffectActionBeginSequenceRender, effectBeginSequenceRender },
	{ kOfxImageEffectActionEndSequenceRender, effectEndSequenceRender },
	{ kOfxImageEffectActionRender, effectRender},
	{ NULL, NULL }
};


/* ------------------------------------------------------------------------- */
/* OpenFX plugin entry points                                                */
/* ------------------------------------------------------------------------- */

static OfxStatus
ofxMain(const char *action,  const void *handle,
        OfxPropertySetHandle inArgs,
	OfxPropertySetHandle outArgs)
{
	/* Try and catch errors */
	try {
		OfxImageEffectHandle effect = (OfxImageEffectHandle) handle;

		for (int i=0; _apiHandlers[i].action; i++)
			if (!strcmp(_apiHandlers[i].action, action))
				return _apiHandlers[i].handler(effect, inArgs, outArgs);

	} catch (const std::bad_alloc&) {
		std::cerr << "[!] OFX Plugin Memory error." << std::endl;
		return kOfxStatErrMemory;
	} catch (const std::exception& e) {
		std::cerr << "[!] OFX Plugin error: " << e.what() << std::endl;
		return kOfxStatErrUnknown;
	} catch (int err) {
		return err;
	} catch ( ... ) {
		std::cerr << "[!] OFX Plugin error" << std::endl;
		return kOfxStatErrUnknown;
	}

	/* Other actions to take the default value */
	return kOfxStatReplyDefault;
}

static void
ofxSetHost(OfxHost *hostStruct)
{
	gHost = hostStruct;
}


/* ------------------------------------------------------------------------- */
/* OpenFX plugin struct and exported func                                    */
/* ------------------------------------------------------------------------- */

static OfxPlugin _plugins[] = {
	[0] = {
		.pluginApi		= kOfxImageEffectPluginApi,
		.apiVersion		= kOfxImageEffectPluginApiVersion,
		.pluginIdentifier	= "be.s47.OfxRobustVideoMatting",
		.pluginVersionMajor	= 0,
		.pluginVersionMinor	= 1,
		.setHost		= ofxSetHost,
		.mainEntry		= ofxMain,
	},
};

EXPORT OfxPlugin *
OfxGetPlugin(int nth)
{
	if ((nth >= 0) && (nth < (int)(sizeof(_plugins) / sizeof(_plugins[0]))))
		return &_plugins[nth];
	return NULL;
}

EXPORT int
OfxGetNumberOfPlugins(void)
{
	return sizeof(_plugins) / sizeof(_plugins[0]);
}
