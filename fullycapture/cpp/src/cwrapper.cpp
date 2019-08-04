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
SMPLModel gSmplModelMale; //the global smplmodel for the cwrapper module
SMPLModel gSmplModelFemale;
SMPLModel gSmplModelNeutral;

extern "C" void load_SMPLModelMale(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz);
extern "C" void load_SMPLModelFemale(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz);
extern "C" void load_SMPLModelNeutral(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz);

extern "C" void load_SMPLPrior(double* prior_A, double* prior_mu, double* prior_weights);
extern "C" void smpl_fit_total_stage1(double* pose, double* coeff, double* trans, double* targetJoint, int reg_type,int gender, bool fit_shape, bool showiter);
extern "C" void setup_fit_options(double body_joint_weight,double finger_weight,double betas_weight, double pose_reg,double disp_reg);




//extern "C" void load_SMPLModel(int *face_,double *J_reg_,double *kintree_,double *J_,double *lbs_w_,double *mu_,double *U_,double * coco_reg_){
extern "C" void load_SMPLModelFemale(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz){
    //reserve space
    gSmplModelFemale.faces_.resize(SMPLModel::NUM_MODEL_FACE, 3);

    gSmplModelFemale.J_reg_.resize(SMPLModel::NUM_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModelFemale.J_reg_.reserve(J_reg_nnz);
    gSmplModelFemale.kintree_table_.resize(2, SMPLModel::NUM_JOINTS);
    

    gSmplModelFemale.W_.resize(SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS);
    gSmplModelFemale.mu_.resize(SMPLModel::NUM_VERTICES * 3, 1);
    gSmplModelFemale.U_.resize(SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS);

    gSmplModelFemale.J_reg_total_.resize(SMPLModel::NUM_TOTAL_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModelFemale.J_reg_total_.reserve(total_reg_nnz);

    //Init with data
    gSmplModelFemale.faces_ = Eigen::Map<Eigen::Matrix<int, SMPLModel::NUM_MODEL_FACE, 3, Eigen::RowMajor>>(face_);
    int* J_reg_row = J_reg_indices;
    int* J_reg_col = J_reg_indices + J_reg_nnz;
    for (int idx = 0; idx < J_reg_nnz; idx++) {
        gSmplModelFemale.J_reg_.insert(J_reg_row[idx], J_reg_col[idx]) = J_reg_val[idx];
    }
    gSmplModelFemale.J_reg_.makeCompressed();

    gSmplModelFemale.kintree_table_ = Eigen::Map<Eigen::Matrix<int, 2, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(kintree_);

    for (int idt = 0; idt < gSmplModelFemale.kintree_table_.cols(); idt++) {
        gSmplModelFemale.id_to_col_[gSmplModelFemale.kintree_table_(1, idt)] = idt;
    }

    for (int idt = 1; idt < gSmplModelFemale.kintree_table_.cols(); idt++) {
        gSmplModelFemale.parent_[idt] = gSmplModelFemale.id_to_col_[gSmplModelFemale.kintree_table_(0, idt)];
    }

    gSmplModelFemale.U_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor>>(U_);
    gSmplModelFemale.mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, 1>>(mu_);
    gSmplModelFemale.J_mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_JOINTS * 3, 1>>(J_mu_);
    gSmplModelFemale.W_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(W_);

    int* total_reg_row = total_reg_ind;
    int* total_reg_col = total_reg_ind + total_reg_nnz;
    for (int idx = 0; idx < total_reg_nnz; idx++) {
        gSmplModelFemale.J_reg_total_.insert(total_reg_row[idx], total_reg_col[idx]) = total_reg_val[idx];
    }
    gSmplModelFemale.J_reg_big_ = Eigen::kroneckerProduct(gSmplModelFemale.J_reg_, Eigen::Matrix<double, 3, 3>::Identity());
    //Init the calculation part
    gSmplModelFemale.dJdc_ = gSmplModelFemale.J_reg_big_ * gSmplModelFemale.U_;

    cout << "Female SMPL Model Loaded" << endl;
}



extern "C" void load_SMPLModelMale(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz){
    //reserve space
    gSmplModelMale.faces_.resize(SMPLModel::NUM_MODEL_FACE, 3);

    gSmplModelMale.J_reg_.resize(SMPLModel::NUM_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModelMale.J_reg_.reserve(J_reg_nnz);
    gSmplModelMale.kintree_table_.resize(2, SMPLModel::NUM_JOINTS);
    

    gSmplModelMale.W_.resize(SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS);
    gSmplModelMale.mu_.resize(SMPLModel::NUM_VERTICES * 3, 1);
    gSmplModelMale.U_.resize(SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS);

    gSmplModelMale.J_reg_total_.resize(SMPLModel::NUM_TOTAL_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModelMale.J_reg_total_.reserve(total_reg_nnz);

    //Init with data
    gSmplModelMale.faces_ = Eigen::Map<Eigen::Matrix<int, SMPLModel::NUM_MODEL_FACE, 3, Eigen::RowMajor>>(face_);
    int* J_reg_row = J_reg_indices;
    int* J_reg_col = J_reg_indices + J_reg_nnz;
    for (int idx = 0; idx < J_reg_nnz; idx++) {
        gSmplModelMale.J_reg_.insert(J_reg_row[idx], J_reg_col[idx]) = J_reg_val[idx];
    }
    gSmplModelMale.J_reg_.makeCompressed();

    gSmplModelMale.kintree_table_ = Eigen::Map<Eigen::Matrix<int, 2, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(kintree_);

    for (int idt = 0; idt < gSmplModelMale.kintree_table_.cols(); idt++) {
        gSmplModelMale.id_to_col_[gSmplModelMale.kintree_table_(1, idt)] = idt;
    }

    for (int idt = 1; idt < gSmplModelMale.kintree_table_.cols(); idt++) {
        gSmplModelMale.parent_[idt] = gSmplModelMale.id_to_col_[gSmplModelMale.kintree_table_(0, idt)];
    }

    gSmplModelMale.U_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor>>(U_);
    gSmplModelMale.mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, 1>>(mu_);
    gSmplModelMale.J_mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_JOINTS * 3, 1>>(J_mu_);
    gSmplModelMale.W_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(W_);

    int* total_reg_row = total_reg_ind;
    int* total_reg_col = total_reg_ind + total_reg_nnz;
    for (int idx = 0; idx < total_reg_nnz; idx++) {
        gSmplModelMale.J_reg_total_.insert(total_reg_row[idx], total_reg_col[idx]) = total_reg_val[idx];
    }
    gSmplModelMale.J_reg_big_ = Eigen::kroneckerProduct(gSmplModelMale.J_reg_, Eigen::Matrix<double, 3, 3>::Identity());
    //Init the calculation part
    gSmplModelMale.dJdc_ = gSmplModelMale.J_reg_big_ * gSmplModelMale.U_;

    cout << "Male SMPL Model Loaded" << endl;
}


extern "C" void load_SMPLModelNeutral(int* face_, double* J_reg_val, int* J_reg_indices, int J_reg_nnz, int* kintree_, double* U_, double* mu_, double* J_mu_, double* W_, double* total_reg_val, int* total_reg_ind, int total_reg_nnz){
    //reserve space
    gSmplModelNeutral.faces_.resize(SMPLModel::NUM_MODEL_FACE, 3);

    gSmplModelNeutral.J_reg_.resize(SMPLModel::NUM_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModelNeutral.J_reg_.reserve(J_reg_nnz);
    gSmplModelNeutral.kintree_table_.resize(2, SMPLModel::NUM_JOINTS);
    

    gSmplModelNeutral.W_.resize(SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS);
    gSmplModelNeutral.mu_.resize(SMPLModel::NUM_VERTICES * 3, 1);
    gSmplModelNeutral.U_.resize(SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS);

    gSmplModelNeutral.J_reg_total_.resize(SMPLModel::NUM_TOTAL_JOINTS, SMPLModel::NUM_VERTICES);
    gSmplModelNeutral.J_reg_total_.reserve(total_reg_nnz);

    //Init with data
    gSmplModelNeutral.faces_ = Eigen::Map<Eigen::Matrix<int, SMPLModel::NUM_MODEL_FACE, 3, Eigen::RowMajor>>(face_);
    int* J_reg_row = J_reg_indices;
    int* J_reg_col = J_reg_indices + J_reg_nnz;
    for (int idx = 0; idx < J_reg_nnz; idx++) {
        gSmplModelNeutral.J_reg_.insert(J_reg_row[idx], J_reg_col[idx]) = J_reg_val[idx];
    }
    gSmplModelNeutral.J_reg_.makeCompressed();

    gSmplModelNeutral.kintree_table_ = Eigen::Map<Eigen::Matrix<int, 2, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(kintree_);

    for (int idt = 0; idt < gSmplModelNeutral.kintree_table_.cols(); idt++) {
        gSmplModelNeutral.id_to_col_[gSmplModelNeutral.kintree_table_(1, idt)] = idt;
    }

    for (int idt = 1; idt < gSmplModelNeutral.kintree_table_.cols(); idt++) {
        gSmplModelNeutral.parent_[idt] = gSmplModelNeutral.id_to_col_[gSmplModelNeutral.kintree_table_(0, idt)];
    }

    gSmplModelNeutral.U_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, SMPLModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor>>(U_);
    gSmplModelNeutral.mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES * 3, 1>>(mu_);
    gSmplModelNeutral.J_mu_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_JOINTS * 3, 1>>(J_mu_);
    gSmplModelNeutral.W_ = Eigen::Map<Eigen::Matrix<double, SMPLModel::NUM_VERTICES, SMPLModel::NUM_JOINTS, Eigen::RowMajor>>(W_);

    int* total_reg_row = total_reg_ind;
    int* total_reg_col = total_reg_ind + total_reg_nnz;
    for (int idx = 0; idx < total_reg_nnz; idx++) {
        gSmplModelNeutral.J_reg_total_.insert(total_reg_row[idx], total_reg_col[idx]) = total_reg_val[idx];
    }
    gSmplModelNeutral.J_reg_big_ = Eigen::kroneckerProduct(gSmplModelNeutral.J_reg_, Eigen::Matrix<double, 3, 3>::Identity());
    //Init the calculation part
    gSmplModelNeutral.dJdc_ = gSmplModelNeutral.J_reg_big_ * gSmplModelNeutral.U_;

    cout << "Neutral SMPL Model Loaded" << endl;
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

extern "C" void smpl_fit_total_stage1(double* pose, double* coeff, double* trans, double* targetJoint, int reg_type,int gender, bool fit_shape, bool showiter){
    SMPLParams frame_params;
    std::copy(trans, trans + 3, frame_params.m_trans.data());
    std::copy(pose, pose + gSmplModelMale.NUM_POSE_PARAMETERS, frame_params.m_pose.data());
    std::copy(coeff, coeff + gSmplModelMale.NUM_SHAPE_COEFFICIENTS, frame_params.m_coeffs.data());
    //get the initial value

    Eigen::MatrixXd TotalJoints;
    int totalJointsNum = gSmplModelMale.NUM_TOTAL_JOINTS;
    
    TotalJoints.resize(4, totalJointsNum); //x,y,z,w
    for (int i = 0; i < totalJointsNum; i++) {
        TotalJoints(0, i) = targetJoint[4 * i + 0];
        TotalJoints(1, i) = targetJoint[4 * i + 1];
        TotalJoints(2, i) = targetJoint[4 * i + 2];
        TotalJoints(3, i) = targetJoint[4 * i + 3];
    }
    if(gender==0)
        SMPL_fit_to_total(gSmplModelFemale, frame_params,gFitOptions,TotalJoints,reg_type, fit_shape, showiter);
    else if(gender==1)
        SMPL_fit_to_total(gSmplModelMale, frame_params,gFitOptions,TotalJoints,reg_type, fit_shape, showiter);
    else
        SMPL_fit_to_total(gSmplModelNeutral,frame_params,gFitOptions,TotalJoints,reg_type,fit_shape,showiter);

    std::copy(frame_params.m_trans.data(), frame_params.m_trans.data() + 3, trans);
    std::copy(frame_params.m_pose.data(), frame_params.m_pose.data() + gSmplModelMale.NUM_POSE_PARAMETERS, pose);
    std::copy(frame_params.m_coeffs.data(), frame_params.m_coeffs.data() + gSmplModelMale.NUM_SHAPE_COEFFICIENTS, coeff);
}