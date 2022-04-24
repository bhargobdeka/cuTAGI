///////////////////////////////////////////////////////////////////////////////
// File:         struct_var.h
// Description:  Header file for struct variable in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 20, 2022
// Updated:      April 21, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <string>
#include <vector>

// NETWORK PROPERTIES
struct LayerLabel {
    int fc = 1;      // Full-connected layer
    int conv = 2;    // Convolutional layer
    int tconv = 21;  // Transpose convolutional layer
    int mp = 3;      // Pooling layer
    int ap = 4;      // Average pooling layer
    int ln = 5;      // Layer normalization layer
    int bn = 6;      // Batch normalization layer
};
struct Network {
    /*Network properties
      Args:
        layers: A vector contains different layers of network architecture
        nodes: Number of hidden units
        kernels: Kernel size fo convolutional layer
        widths: Width of image
        heights: Heights of image
        filters: Number of filters i.e. depth of image for each layer
        activations: Activation function
        pads: Padding that applied to image
        pad_types: Type of padding
        num_weights: Number of weights
        num_biases: Number of biases
        num_weights_sc: Number of weights for residual network
        num_biases_sc: Number of biases for residual network
        similar_layers: Index of layers having the smilar indices
        layer_names: Conventional name for each layer
        init_method: Initalization method e.g. He and Xavier
        gain_w: Gain for weight parameters when initializing
        gain_b: Gain for biases parameters when initializing
        w_pos: Weights weight position for each layer in the vector of weights
        b_pos: Biases position for each layer in the vector of biases
        w_pos_sc: Weights position for each layer in the vector of weights for
                  resnet
        b_pos_sc: Biases position for each layer in the vector of biases
                  for resnet
        z_pos: Hidden state's position for each layers in the hidden state
              vector
        sc_pos: Position of the shortcut's hidden state in the shortcut's hidden
                state vector
        ra_pos: Statistical mean and variance position for the
                normalization layer
        overlap: Binary vector that indicates if the kernel size creates an
                 overlap pass through images

        ***********************************************************************
        Fmwa_1_pos: Position of weight indices for mean product WA
        Fmwa_2_pos: Position of either hidden-states and activaition-units
                    indices when perform TAGI's forward pass
        FCzwa_1_pos: Position of weight indices for covariance Z|WA
        FCzwa_2_pos: Position of activation indices for covariance Z|WA
        Szz_ud_pos: Postision of next hidden state indices for covariance Z|Z+
        FCwz_2_pos: Position of activation indices for covariance W|Z+
        Swz_ud_pos: Position of hidden state (Z+) indices for covariance Z|Z+
        row_zw: Number of rows of weight indices for covariance Z|WA
        col_z_ud: Number of columns of next hidden state indices for covariance
                  Z|Z+
        Fmwa_1_col: Number of columns of weight indices for mean product WA
        FCzwa_1_col:Number of column of weight indices for covariance Z|WA

        ***********************************************************************
        batch_size: Number of batches of data
        n_state: Total number of states
        n_max_state: Maximum number of states amongst layers
        n_state_sc: Number of states for residual network
        n_ra: Number of statistical mean and variances for the normalization
        layer init_sc: First shortcut layer for residual network
        n_nye: Number of observation for hierarchical softmax
        last_backward_layer: Index of last layer whose hidden states are updated
        sigma_v: Observation noise
        alpha: alpha in leakylu
        epsilon: Constant for normalization layer to avoid zero-division
        ra_mt: Momentum for the normalization layer
        is_output_ud: Whether or not to update output layer
        is_idx_ud: Wheher or not to update only hidden units in the output
                   layers

    NOTE*: sc means the shortcut for residual network.
    */
    std::vector<int> layers, nodes, kernels, strides, widths, heights, filters,
        activations;
    std::vector<int> pads, pad_types, shortcuts, num_weights, num_biases;
    std::vector<int> num_weights_sc, num_biases_sc, similar_layers;
    LayerLabel layer_names;
    std::string init_method;
    std::vector<int> gain_w, gain_b, w_pos, b_pos, w_sc_pos, b_sc_pos;
    std::vector<int> z_pos, sc_pos, ra_pos, overlap;

    // Position for each layer in index vector
    std::vector<int> Fmwa_1_pos, Fmwa_2_pos, FCzwa_1_pos, FCzwa_2_pos,
        Szz_ud_pos, pooling_pos;
    std::vector<int> FCwz_2_pos, Swz_ud_pos, Fmwa_2_sc_pos, FCzwa_1_sc_pos;
    std::vector<int> FCzwa_2_sc_pos, Szz_ud_sc_pos, row_zw, col_z_ud, row_w_sc;
    std::vector<int> col_z_sc;
    std::vector<int> Fmwa_1_col, FCzwa_1_col;

    int batch_size = 1;
    int n_state = 1;
    int n_max_state = 1;
    int n_state_sc = 1;
    int n_ra = 1;
    int init_sc = -1;
    int nye = 0;
    int last_backward_layer = 1;
    float sigma_v = 0.0;
    float alpha = 0.1;
    float epsilon = 0.0001f;
    float ra_mt = 0.9;
    bool is_output_ud = true;
    bool is_idx_ud = false;
};

// NETWORK STATE
struct NetState {
    /* Network's hidden states
       Args:
        mz: Mean of hidden states
        Sz: Variance of hidden states
        ma: Mean of activation units
        Sa: Variance of activation units
        J: Jacobian matrix
        msc: Mean of the shortcut's hidden states
        Ssc: Varaince of the short's hidden states
        mdsc: Mean of the delta shortcut's hidden states
        Sdsc: Varaince of the delta short's hidden states
        mra: Statistical mean for the normalzation layers
        Sra: Statistical variance for the normalzation layers
    */
    std::vector<float> mz, Sz, ma, Sa, J, msc, Ssc, mdsc, Sdsc;
    std::vector<float> mra, Sra;
};

// NETWORK PARAMETERS
struct Param {
    /* Network's hidden states
       Args:
        mw: Mean of weights
        Sw: Variance of weights
        mb: Mean of the biases
        Sb: Variance of the biases
        mw_sc: Mean of weights for residual network
        Sw_sc: Variance of weights for residual network
        mb_sc: Mean of biases for residual network
        Sb_sc: Variance of biases for residual network
    */

    std::vector<float> mw, Sw, mb, Sb, mw_sc, Sw_sc, mb_sc, Sb_sc;
};

// NETWORK INDICES
struct RefIndexOut {
    std::vector<int> ref, base_idx, pad_pos;
    int w, h;
};
struct ConvIndexOut {
    std::vector<int> Fmwa_2_idx, FCzwa_1_idx, FCzwa_2_idx, Szz_ud_idx;
    int w, h;
};
struct PoolIndex {
    std::vector<int> pool_idx, Szz_ud_idx;
    int w, h;
};
struct TconvIndexOut {
    std::vector<int> FCwz_2_idx, Swz_ud_idx, FCzwa_1_idx, Szz_ud_idx;
    int w_wz, h_wz, w_zz, h_zz;
};

struct IndexOut {
    /* Network's hidden states
       Args:
        Fmwa_1: Weight indices for mean product WA
        Fmwa_2: Activation indices for mean product WA
        FCzwa_1: Weight indices for covariance Z|WA
        FCzwa_2: Activaiton indices for covariance Z|WA
        Szz_ud: Next hidden state indices for covariance Z|Z+
        pooling: Pooling index
        FCwz_2: Activation indices for covariance W|Z+
        Swz_ud: Hidden state (Z+) indices for covariance Z|Z+

    NOTE*: The extension _sc means shortcut i.e. the same indices meaning for
    the residual network
    */
    std::vector<int> Fmwa_1, Fmwa_2, FCzwa_1, FCzwa_2, Szz_ud, pooling, FCwz_2,
        Swz_ud;
    std::vector<int> Fmwa_2_sc, FCzwa_1_sc, FCzwa_2_sc, Szz_ud_sc;
};

// IMAGE DATA
struct ImageData {
    /* Image database's format
       Args:
        images: Images stored as a giant vector
        obs_label: Converted observation from the labels
        obs_idx: Observation indices assigned to each label
        labels: Raw labels

    NOTE*: In the context of TAGI, we uses the hierarchical softmax for the
    classification problem. So, we convert the raw lable for the label fictive
    observation.
    */
    std::vector<float> images;
    std::vector<float> obs_label;
    std::vector<int> obs_idx;
    std::vector<int> labels;
    int num_data;
};

// HIERARCHICAL SOFTMAX
struct HrSoftmax {
    /* Hierarchical softmax
       Args:
        obs: A fictive observation \in [-1, 1]
        idx: Indices assigned to each label
        n_obs: Number of indices for each label
        len: Length of an observation e.g 10 labels -> len(obs) = 11
    */
    std::vector<float> obs;
    std::vector<int> idx;
    int n_obs, len;
};

// DIRECTORIES
struct SavePath {
    /* Different directories to store the results
       Args:
        curr_path: Current path of the workspacefolder
        saved_param_path: Directory to store the weights and biases
        saved_inference_path: Directory to store the inference results
        debug_path: Directory to store network's indices, parameters, inference
                    results that then uses to verify with test units.
    */
    std::string curr_path, saved_param_path, saved_inference_path, debug_path;
};