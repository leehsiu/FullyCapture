#ifndef POSE_TO_TRANSFORMS_H
#define POSE_TO_TRANSFORMS_H

#include "smplModel.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <iostream>
#include <chrono>

void AngleAxisToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns=SMPLModel::NUM_JOINTS * 3);
void EulerAnglesToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns=SMPLModel::NUM_JOINTS * 3);
void Product_Derivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                        const double* const dB_data, double* dAB_data, const int B_col=3);
void SparseProductDerivative(const double* const A_data, const double* const dA_data, const double* const B_data,
                             const double* const dB_data, const int colIndex,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns=SMPLModel::NUM_JOINTS * 3);
void SparseProductDerivative(const double* const dA_data, const double* const B_data,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns=SMPLModel::NUM_JOINTS * 3);
void SparseProductDerivativeConstA(const double* const A_data, const double* const dB_data,
                             const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns=SMPLModel::NUM_JOINTS * 3);
void SparseAdd(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns=SMPLModel::NUM_JOINTS * 3);
void SparseSubtract(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns=SMPLModel::NUM_JOINTS * 3);


class PoseToTransform_SMPLFull_withDiff: public ceres::CostFunction{
public:
	PoseToTransform_SMPLFull_withDiff(const SMPLModel &mod, const std::array<std::vector<int>, SMPLModel::NUM_JOINTS>& parentIndices):
		mod_(mod), mParentIndices(parentIndices)
	{
		CostFunction::set_num_residuals(3 * 5 * SMPLModel::NUM_JOINTS);
		auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
		parameter_block_sizes->clear();
		parameter_block_sizes->push_back(SMPLModel::NUM_JOINTS * 3); // SMPL Pose  
		parameter_block_sizes->push_back(SMPLModel::NUM_JOINTS * 3); // SMPL Joint  
	}
	virtual ~PoseToTransform_SMPLFull_withDiff() {}

	virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const;

	const SMPLModel &mod_;
	const std::array<std::vector<int>, SMPLModel::NUM_JOINTS>& mParentIndices;
};




#endif