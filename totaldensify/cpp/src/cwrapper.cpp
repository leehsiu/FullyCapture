// As a wrapper to Python

#include"mainFit.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <FreeImage.h>
#include <GL/freeglut.h>
#include <assert.h>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>
#include"smplModel.h"
using namespace std;

SMPLFitOptions gFitOptions;
SMPLModel gSmplModel; //the global smplmodel for the cwrapper module

extern "C" void load_SMPLModel(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz);
extern "C" void load_SMPLPrior(double* prior_A, double* prior_mu, double* prior_weights);
extern "C" void smpl_fit_total_stage1(double* pose, double* coeff, double* trans, double* targetJoint, int reg_type, bool fit_shape, bool showiter);
extern "C" void setup_fit_options(double body_joint_weight,double finger_weight,double betas_weight, double pose_reg,double disp_reg);

//extern "C" void load_SMPLModel(int *face_,double *J_reg_,double *kintree_,double *J_,double *lbs_w_,double *mu_,double *U_,double * coco_reg_){

extern "C" void load_SMPLModel(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz){
    //reserve space
    gSmplModel.faces_.resize(SMPLModel::NUM_MODEL_FACE, 3);

    gSmplModel.J_reg_.resize(SMPLModel::NUM_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModel.J_reg_.reserve(J_reg_nnz);
    gSmplModel.kintree_table_.resize(2, SMPLModel::NUM_JOINTS);
    

    gSmplModel.W_.resize(SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS);
    gSmplModel.mu_.resize(SMPLModel::NUM_VERTICES * 3, 1);
    gSmplModel.U_.resize(SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS);

    gSmplModel.J_reg_total_.resize(SMPLModel::NUM_TOTAL_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModel.J_reg_total_.reserve(total_reg_nnz);

    //Init with data
    gSmplModel.faces_ = Eigen::Map<Eigen::Matrix<int, SMPLModel::NUM_MODEL_FACE, 3, Eigen::RowMajor>>(face_);
    int* J_reg_row = J_reg_indices;
    int* J_reg_col = J_reg_indices + J_reg_nnz;
    for (int idx = 0; idx < J_reg_nnz; idx++) {
        gSmplModel.J_reg_.insert(J_reg_row[idx], J_reg_col[idx]) = J_reg_val[idx];
    }
    gSmplModel.J_reg_.makeCompressed();

    gSmplModel.kintree_table_ = Eigen::Map<Eigen::Matrix<int, 2, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(kintree_);

    for (int idt = 0; idt < gSmplModel.kintree_table_.cols(); idt++) {
        gSmplModel.id_to_col_[gSmplModel.kintree_table_(1, idt)] = idt;
    }

    for (int idt = 1; idt < gSmplModel.kintree_table_.cols(); idt++) {
        gSmplModel.parent_[idt] = gSmplModel.id_to_col_[gSmplModel.kintree_table_(0, idt)];
    }

    gSmplModel.U_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor>>(U_);
    gSmplModel.mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, 1>>(mu_);
    gSmplModel.J_mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_JOINTS * 3, 1>>(J_mu_);
    gSmplModel.W_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(W_);

    int* total_reg_row = total_reg_ind;
    int* total_reg_col = total_reg_ind + total_reg_nnz;
    for (int idx = 0; idx < total_reg_nnz; idx++) {
        gSmplModel.J_reg_total_.insert(total_reg_row[idx], total_reg_col[idx]) = total_reg_val[idx];
    }
    gSmplModel.J_reg_big_ = Eigen::kroneckerProduct(gSmplModel.J_reg_, Eigen::Matrix<double, 3, 3>::Identity());
    //Init the calculation part
    gSmplModel.dJdc_ = gSmplModel.J_reg_big_ * gSmplModel.U_;

    cout << "SMPL Model Loaded" << endl;
}


extern "C" void setup_fit_options(double body_joint_weight,double finger_weight,double betas_weight, double pose_reg,double disp_reg){
    gFitOptions.body_joints_weight = body_joint_weight;
    gFitOptions.hand_joints_weight = finger_weight;
    gFitOptions.coeff_weight = betas_weight;
    gFitOptions.reg_pose_weight = pose_reg;
    gFitOptions.disp_pose_weight = disp_reg;

}
extern "C" void load_SMPLPrior(double* prior_A, double* prior_mu, double* prior_weights){
    
    
    //! temporally removed all priors
    // gSmplModel.pose_prior_A.resize(8, 69 * 69);
    // gSmplModel.pose_prior_mu.resize(8, 69);
    // gSmplModel.pose_prior_b.resize(8, 1);

    // gSmplModel.pose_prior_A = Eigen::Map<Eigen::Matrix<double, 8, 69 * 69, Eigen::RowMajor>>(prior_A);
    // gSmplModel.pose_prior_mu = Eigen::Map<Eigen::Matrix<double, 8, 69, Eigen::RowMajor>>(prior_mu);
    // gSmplModel.pose_prior_b = Eigen::Map<Eigen::Matrix<double, 8, 1>>(prior_weights);
    return;
}

extern "C" void smpl_fit_total_stage1(double* pose, double* coeff, double* trans, double* targetJoint, int reg_type, bool fit_shape, bool showiter){
    SMPLParams frame_params;
    std::copy(trans, trans + 3, frame_params.m_trans.data());
    std::copy(pose, pose + gSmplModel.NUM_POSE_PARAMETERS, frame_params.m_pose.data());
    std::copy(coeff, coeff + gSmplModel.NUM_SHAPE_COEFFICIENTS, frame_params.m_coeffs.data());
    //get the initial value

    Eigen::MatrixXd TotalJoints;
    int totalJointsNum = gSmplModel.NUM_TOTAL_JOINTS;
    
    TotalJoints.resize(4, totalJointsNum); //x,y,z,w
    for (int i = 0; i < totalJointsNum; i++) {
        TotalJoints(0, i) = targetJoint[4 * i + 0];
        TotalJoints(1, i) = targetJoint[4 * i + 1];
        TotalJoints(2, i) = targetJoint[4 * i + 2];
        TotalJoints(3, i) = targetJoint[4 * i + 3];
    }

    SMPL_fit_to_total(gSmplModel, frame_params,gFitOptions,TotalJoints,reg_type, fit_shape, showiter);
    std::copy(frame_params.m_trans.data(), frame_params.m_trans.data() + 3, trans);
    std::copy(frame_params.m_pose.data(), frame_params.m_pose.data() + gSmplModel.NUM_POSE_PARAMETERS, pose);
    std::copy(frame_params.m_coeffs.data(), frame_params.m_coeffs.data() + gSmplModel.NUM_SHAPE_COEFFICIENTS, coeff);
}