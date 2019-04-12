#ifndef SMPL_MAIN_FIT
#define SMPL_MAIN_FIT
#include "smplModel.h"
#include "ceres/ceres.h"

struct SMPLFitOptions{
	int reg_type;
    double body_joints_weight;
    double hand_joints_weight;
	double coeff_weight;
	double reg_pose_weight;
    double disp_pose_weight;
};
//store the options
struct SMPLFitData {
    SMPLFitData(SMPLModel& smplmodel, Eigen::MatrixXd& Joints)
        : smplModel(smplmodel)
        , totalJoints(Joints)
    {
        ;
    }
    const SMPLModel& smplModel;
    const Eigen::MatrixXd& totalJoints;
};


void SetSolverOptions(ceres::Solver::Options *options);

void SMPL_fit_to_total(SMPLModel& smplModel,
    SMPLParams& frame_param,
    SMPLFitOptions &fit_options,
    Eigen::MatrixXd& BodyJoints,
    uint reg_type,
    bool fit_shape,
    bool showIter);
#endif