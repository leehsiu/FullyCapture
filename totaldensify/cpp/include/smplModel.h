#ifndef SIMPLE_H
#define SIMPLE_H

#include "cv.h"
#include <Eigen/Sparse>
#include <utility>
#include <vector>
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdr;

struct SMPLModel{
    static const int NUM_SHAPE_COEFFICIENTS = 10;
    static const int NUM_VERTICES = 6890;
    static const int NUM_JOINTS = 52;
    static const int NUM_POSE_PARAMETERS = NUM_JOINTS * 3;
    static const int NUM_TOTAL_JOINTS = 65;
    static const int NUM_MODEL_FACE = 13776;

    // Template vertices (vector) <NUM_VERTICES*3, 1>
    Eigen::Matrix<double, Eigen::Dynamic, 1> mu_;

    // Shape basis, <NUM_FACE_POINTS*3, NUM_COEFFICIENTS>
    Eigen::Matrix<double, Eigen::Dynamic, NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor> U_;

    // LBS weights,
    Eigen::Matrix<double, Eigen::Dynamic, NUM_JOINTS, Eigen::RowMajor> W_;

    // J_mu_ = J_reg_big_ * mu_
    Eigen::Matrix<double, NUM_JOINTS * 3, 1> J_mu_;

    // dJdc = J_reg_big_ * U_
    Eigen::Matrix<double, Eigen::Dynamic, NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor> dJdc_;

    // Joint regressor, <NUM_JOINTS, NUM_VERTICES>
    Eigen::SparseMatrix<double> J_reg_;

    // Joint regressor, <NUM_JOINTS*, NUM_VERTICES*3>  kron(J_reg_, eye(3))
    Eigen::SparseMatrix<double, Eigen::RowMajor> J_reg_big_;
    Eigen::SparseMatrix<double, Eigen::ColMajor> J_reg_big_col_;

    // Pose regressor, <NUM_VERTICES*3, (NUM_JOINTS-1)*9>
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pose_reg_;

    // Final regressor for LSP (Human3.6M) 14 * NUM_JOINTS
    Eigen::SparseMatrix<double, Eigen::RowMajor> J_reg_lsp_;
    // Final regressor for COCO 19 * NUM_JOINTS
    Eigen::SparseMatrix<double> J_reg_coco_;

    // Final regressor for Total 65 joints
    Eigen::SparseMatrix<double> J_reg_total_;

    // Shape coefficient weights
    Eigen::Matrix<double, Eigen::Dynamic, 1> d_;

    // Triangle faces
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> faces_;

    // Triangle UV map
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> uv; //UV texture coordinate (same number as vertex)

    // Kinematic tree
    Eigen::Matrix<int, 2, Eigen::Dynamic> kintree_table_;
    int parent_[NUM_JOINTS];
    int id_to_col_[NUM_JOINTS];

    //Merging part (hand and face) together

    // A model is fully specified by its coefficients, pose, and a translation
    Eigen::Matrix<double, NUM_SHAPE_COEFFICIENTS, 1> coeffs;
    Eigen::Matrix<double, NUM_JOINTS, 3, Eigen::RowMajor> pose;
    Eigen::Vector3d t;

    bool bInit;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct SMPLParams {
    // A model is fully specified by its coefficients, pose, and a translation
    Eigen::Matrix<double, 52, 3, Eigen::RowMajor> m_pose; //52
    Eigen::Matrix<double, 10, 1> m_coeffs; //30 ==TotalModel::NUM_SHAPE_COEFFICIENTS
    Eigen::Vector3d m_trans;

    SMPLParams()
    {
        m_pose.setZero();
        m_coeffs.setZero();
        m_trans.setZero();
    }
};

#endif