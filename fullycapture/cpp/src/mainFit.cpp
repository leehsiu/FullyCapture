#include "smplCostFunc.h"
#include "smplFK.h"
#include "smplModel.h"
#include "priorCostFunc.h"
#include "ceres/normal_prior.h"
#include <chrono>
#include <iostream>

void SetSolverOptions(ceres::Solver::Options* options)
{
    CHECK(StringToLinearSolverType("sparse_normal_cholesky",
        &options->linear_solver_type));
    CHECK(StringToPreconditionerType("jacobi",
        &options->preconditioner_type));
    options->num_linear_solver_threads = 4;
    options->max_num_iterations = 15;
    options->num_threads = 10;
    options->dynamic_sparsity = true;
    options->use_nonmonotonic_steps = true;
    CHECK(StringToTrustRegionStrategyType("levenberg_marquardt",
        &options->trust_region_strategy_type));
}

void SMPL_fit_to_total(SMPLModel& smplModel,
    SMPLParams& frame_param,
    SMPLFitOptions &fit_options,
    Eigen::MatrixXd& BodyJoints,
    uint reg_type,
    bool fit_shape,
    bool showIter)
{
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    //save the init values firstly.
    Eigen::Matrix<double, 52*3, 1> init_mu;
    double * init_pose =new double[52*3];
    std::copy(frame_param.m_pose.data(),frame_param.m_pose.data() + smplModel.NUM_POSE_PARAMETERS,init_pose);
    init_mu = Eigen::Map<Eigen::Matrix<double,52*3,1>>(init_pose);


    SetSolverOptions(&options);
    options.function_tolerance = 1e-7;
    options.max_num_iterations = 100;
    options.use_nonmonotonic_steps = true;
    options.num_linear_solver_threads = 10;
    options.minimizer_progress_to_stdout = true;
    if (!showIter)
        options.logging_type = ceres::SILENT;
    

    SMPLFitData fit_data(smplModel,BodyJoints);

    //*0. The main error, 3D euclidiean distance
    SMPLFullCost* smpl_cost = new SMPLFullCost(fit_data,fit_options);
    ceres::LossFunction* smpl_loss = new ceres::ScaledLoss(NULL,1e1,ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(smpl_cost,
        smpl_loss,
        frame_param.m_trans.data(),
        frame_param.m_pose.data(),
        frame_param.m_coeffs.data());

    
    //*1. Shape Prior, zero-mean normal prior
    CoeffsParameterNormDiff_SMPL* cost_prior_coeffs = new CoeffsParameterNormDiff_SMPL(SMPLModel::NUM_SHAPE_COEFFICIENTS);
    ceres::LossFunction* loss_prior_coeffs = new ceres::ScaledLoss(NULL,
        fit_options.coeff_weight,
        ceres::TAKE_OWNERSHIP);

    ceres::LossFunctionWrapper* loss_prior_coeffs_wrapper = new ceres::LossFunctionWrapper(loss_prior_coeffs, ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(cost_prior_coeffs,
        loss_prior_coeffs_wrapper,
        frame_param.m_coeffs.data());


    //*2. Pose Prior, zero-mean regulize
    SMPLPosePrior_withDiff* cost_reg_pose = new SMPLPosePrior_withDiff(SMPLModel::NUM_POSE_PARAMETERS);

    ceres::LossFunction* loss_reg_pose = new ceres::ScaledLoss(NULL,
        fit_options.reg_pose_weight,
        ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(cost_reg_pose,
        loss_reg_pose,
        frame_param.m_pose.data());



    //*3. Displacements from init value
    //* cost  = |A(x-mu)|^2
    
    Eigen::MatrixXd init_A(52*3,52*3);
    init_A.setZero();
    for(int diag=3;diag<22*3;diag++){
        init_A(diag,diag) = 1;
    }
    for(int diag=22*3;diag<52*3;diag++){
        init_A(diag,diag) = 0.2;
    }



    ceres::CostFunction *normal_cost = new ceres::NormalPrior(init_A,init_mu);
    ceres::LossFunction *normal_loss = new ceres::ScaledLoss(NULL,fit_options.disp_pose_weight,ceres::TAKE_OWNERSHIP);
    problem.AddResidualBlock(normal_cost,normal_loss,frame_param.m_pose.data());

    //calc the normal error as init_A*|pose-init_mu|

    //*3. Mixture Gaussian Prior, disabled for non-prior fitting
    //temporally disabled for debuggin.
    // ceres::LossFunctionWrapper* pose_gmm_loss_wrapper[8];
    // ceres::LossFunction* pose_gmm_loss[8];
    // ceres::CostFunction* pose_gmm_cost[8];

    // double gmm_cost_w[4] = { 1e-1, 1e-2, 5 * 1e-3, 1e-4 };
    // double prior_w = gmm_cost_w[0];
    // for (int i = 0; i < 8; i++) {
    //     Eigen::Matrix<double, 52*3, 52*3> prior_A;
    //     Eigen::Matrix<double, 52*3, 1> prior_mu;
    //     prior_A.setZero();
    //     prior_mu.setZero();

    //     prior_A.block<69, 69>(3, 3) = Eigen::Map<Eigen::Matrix<double, 69, 69, Eigen::RowMajor>>(smplModel.pose_prior_A.row(i).data());
    //     prior_mu.block<69, 1>(3, 0) = Eigen::Map<Eigen::Matrix<double, 69, 1>>(smplModel.pose_prior_mu.data() + 69 * i);
    //     double w = smplModel.pose_prior_b(i);

    //     pose_gmm_cost[i] = new ceres::NormalPrior(prior_A, - prior_mu);
    //     pose_gmm_loss[i] = new ceres::ScaledLoss(NULL, prior_w * w, ceres::TAKE_OWNERSHIP);
    //     pose_gmm_loss_wrapper[i] = new ceres::LossFunctionWrapper(pose_gmm_loss[i], ceres::TAKE_OWNERSHIP);
    //     problem.AddResidualBlock(pose_gmm_cost[i], pose_gmm_loss_wrapper[i], frame_param.pose.data());
    // }

    //calculate the errors and use softmax to put weight on each part.

    // if (!fit_) {
    //     problem.SetParameterBlockConstant(frame_param.coeffs.data());
    // }

    if (showIter)
        std::cout << "Problem Created:start to solve" << std::endl;
    smpl_cost->toggle_activate(false,false,false);
    smpl_cost->toggle_rigid_body(true);
    ceres::Solve(options, &problem, &summary);

    smpl_cost->toggle_activate(true,false,false);
    smpl_cost->toggle_rigid_body(false);
    ceres::Solve(options, &problem, &summary);

    smpl_cost->toggle_activate(true,true,false);
    smpl_cost->toggle_rigid_body(false);
    ceres::Solve(options, &problem, &summary);

    smpl_cost->toggle_activate(true,true,true);
    smpl_cost->toggle_rigid_body(false);
    ceres::Solve(options, &problem, &summary);
    if (showIter)
        std::cout << summary.FullReport() << std::endl;
}