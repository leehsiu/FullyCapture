# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Detectron config system.

This file specifies default config options for Detectron. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.

Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See tools/{train,test}_net.py for example code that uses merge_cfg_from_file
 - See configs/*/*.yaml for example config files

Detectron supports a lot of different model types, each of which has a lot of
different options. The result is a HUGE set of configuration options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from future.utils import iteritems
from past.builtins import basestring
import copy
import logging
import numpy as np
import os
import os.path as osp
import yaml

from totaldensify.utils.collections import AttrDict

logger = logging.getLogger(__name__)

__C = AttrDict()
# Consumers can get config by:
#   from detectron.core.config import cfg
cfg = __C


# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.CAPTURE = AttrDict()

# Initialize network with weights from this .pkl file
__C.CAPTURE.PRIOR = b''

__C.CAPTURE.BODY_WEIGHT = 1.0

__C.CAPTURE.HAND_WEIGHT = 0.8

__C.CAPTURE.SHAPE_REG = 1e2

__C.CAPTURE.POSE_REG = 1e2

__C.CAPTURE.MOTION_REG = 1e2

__C.CAPTURE.MODEL_FEMALE = b''
__C.CAPTURE.MODEL_MALE = b''
__C.CAPTURE.MODEL_NEUTRAL= b''

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set(
    {
        'FINAL_MSG',
        'MODEL.DILATION',
        'ROOT_GPU_ID',
        'RPN.ON',
        'TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED',
        'TRAIN.DROPOUT',
        'USE_GPU_NMS',
        'TEST.NUM_TEST_IMAGES',
    }
)


# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'MODEL.PS_GRID_SIZE': 'RFCN.PS_GRID_SIZE',
    'MODEL.ROI_HEAD': 'FAST_RCNN.ROI_BOX_HEAD',
    'MRCNN.MASK_HEAD_NAME': 'MRCNN.ROI_MASK_HEAD',
    'TRAIN.DATASET': (
        'TRAIN.DATASETS',
        "Also convert to a tuple, e.g., " +
        "'coco_2014_train' -> ('coco_2014_train',) or " +
        "'coco_2014_train:coco_2014_valminusminival' -> " +
        "('coco_2014_train', 'coco_2014_valminusminival')"
    ),
    'TRAIN.PROPOSAL_FILE': (
        'TRAIN.PROPOSAL_FILES',
        "Also convert to a tuple, e.g., " +
        "'path/to/file' -> ('path/to/file',) or " +
        "'path/to/file1:path/to/file2' -> " +
        "('path/to/file1', 'path/to/file2')"
    ),
    'TEST.SCALES': (
        'TEST.SCALE',
        "Also convert from a tuple, e.g. (600, ), " +
        "to a integer, e.g. 600."
    ),
    'TEST.DATASET': (
        'TEST.DATASETS',
        "Also convert from a string, e.g 'coco_2014_minival', " +
        "to a tuple, e.g. ('coco_2014_minival', )."
    ),
    'TEST.PROPOSAL_FILE': (
        'TEST.PROPOSAL_FILES',
        "Also convert from a string, e.g. '/path/to/props.pkl', " +
        "to a tuple, e.g. ('/path/to/props.pkl', )."
    ),
}


# ---------------------------------------------------------------------------- #
# Renamed modules
# If a module containing a data structure used in the config (e.g. AttrDict)
# is renamed/moved and you don't want to break loading of existing yaml configs
# (e.g. from weights files) you can specify the renamed module below.
# ---------------------------------------------------------------------------- #
_RENAMED_MODULES = {
    'utils.collections': 'detectron.utils.collections',
}


def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if make_immutable:
        cfg.immutable(True)




def get_output_dir(datasets, training=True):
    """Get the output directory determined by the current global config."""
    assert isinstance(datasets, (tuple, list, basestring)), \
        'datasets argument must be of type tuple, list or string'
    is_string = isinstance(datasets, basestring)
    dataset_name = datasets if is_string else '_'.join(datasets)
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset-name>/<model-type>/
    outdir = osp.join(__C.OUTPUT_DIR, tag, dataset_name, __C.MODEL.TYPE)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir


def load_cfg(cfg_to_load):
    """Wrapper around yaml.load used for maintaining backward compatibility"""
    assert isinstance(cfg_to_load, (file, basestring)), \
        'Expected {} or {} got {}'.format(file, basestring, type(cfg_to_load))
    if isinstance(cfg_to_load, file):
        cfg_to_load = ''.join(cfg_to_load.readlines())
    if isinstance(cfg_to_load, basestring):
        for old_module, new_module in iteritems(_RENAMED_MODULES):
            # yaml object encoding: !!python/object/new:<module>.<object>
            old_module, new_module = 'new:' + old_module, 'new:' + new_module
            cfg_to_load = cfg_to_load.replace(old_module, new_module)
    return yaml.load(cfg_to_load)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(load_cfg(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        logger.warn(
            'Deprecated config key (ignoring): {}'.format(full_key)
        )
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
        format(full_key, new_key, msg)
    )


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
