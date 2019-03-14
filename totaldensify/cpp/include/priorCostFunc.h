#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "smplModel.h"

#include <unsupported/Eigen/KroneckerProduct>
#include <cassert>
#include <chrono>
#define SMPL_VIS_SCALING 100.0f

struct CoeffsParameterNorm {
	CoeffsParameterNorm(int num_parameters)
		: num_parameters_(num_parameters) {}

	template <typename T>
	inline bool operator()(const T* const p,
		T* residuals) const {
			for (int i = 0; i < num_parameters_; i++) 
			{
				residuals[i] = T(1.0)*p[i];
			}
		return true;
	}
	const double num_parameters_;
};

class CoeffsParameterNormDiff: public ceres::CostFunction
{
public:
	CoeffsParameterNormDiff(int num_parameters): num_parameters_(num_parameters)
	{
		assert(num_parameters == SMPLModel::NUM_SHAPE_COEFFICIENTS); // Adam model
		CostFunction::set_num_residuals(num_parameters_);
		CostFunction::mutable_parameter_block_sizes()->clear();
		CostFunction::mutable_parameter_block_sizes()->push_back(num_parameters_);
	}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		double const* p = parameters[0];
		std::copy(p, p + num_parameters_, residuals);
		if (jacobians)
		{
			if (jacobians[0])
			{
				double* jac = jacobians[0];
				std::fill(jac, jac + num_parameters_ * num_parameters_, 0);
				for (int i = 0; i < num_parameters_; i++)
					jac[i * num_parameters_ + i] = 1.0;
			}
		}
		return true;
	}
private:
	const int num_parameters_;
};




class CoeffsParameterNormDiff_SMPL: public ceres::CostFunction
{
public:
	CoeffsParameterNormDiff_SMPL(int num_parameters): num_parameters_(num_parameters)
	{
		assert(num_parameters == SMPLModel::NUM_SHAPE_COEFFICIENTS); // Adam model
		CostFunction::set_num_residuals(num_parameters_);
		CostFunction::mutable_parameter_block_sizes()->clear();
		CostFunction::mutable_parameter_block_sizes()->push_back(num_parameters_);
	}
	virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
	{
		double const* p = parameters[0];
		std::copy(p, p + num_parameters_, residuals);
		if (jacobians)
		{
			if (jacobians[0])
			{
				double* jac = jacobians[0];
				std::fill(jac, jac + num_parameters_ * num_parameters_, 0);
				for (int i = 0; i < num_parameters_; i++)
					jac[i * num_parameters_ + i] = 1.0;
			}
		}
		return true;
	}
private:
	const int num_parameters_;
};

struct CoeffsParameterLogNorm {
	CoeffsParameterLogNorm(int num_parameters)
		: num_parameters_(num_parameters) {}

	template <typename T>
	inline bool operator()(const T* const p,
		T* residuals) const {
			for (int i = 0; i < num_parameters_; i++) 
			{
				residuals[i] = T(1.0)*log(p[i]);
			}
		return true;
	}
	const double num_parameters_;
};
struct AdamBodyPoseParamPrior {  // used by refitting
	AdamBodyPoseParamPrior(int num_parameters)
		: num_parameters_(num_parameters) {}

	// <Residuals, block1 params, block2 params>  = <2, 6, nIntrinsicParams, 3>
	template <typename T>
	inline bool operator()(const T* const p,
		T* residuals) const {


		//Put stronger prior for spine body joints
		for (int i = 0; i < num_parameters_; i++)
		{
			if (i >= 0 && i < 3)
			{
				residuals[i] = T(3)*p[i];
			}
			else if ((i >= 9 && i < 12) || (i >= 18 && i < 21) || (i >= 27 && i < 30))
			{
				residuals[i] = T(3)*p[i];
			}
			else if ((i >= 42 && i < 45) || (i >= 39 && i < 41))
				residuals[i] = T(2)*p[i];
			else if (i >= 54 && i < 60)		//18, 19 (elbows)
			{
				if (i == 54 || i == 57)		//twist 
					residuals[i] = T(1)*p[i];
				else
					residuals[i] = T(0.1)*p[i];
			}
			else if (i >= 60 && i < 66)		//20, 21 (wrist)
			{
				if (i == 60 || i == 63)		//twist of wrist
					residuals[i] = T(1)*p[i];
				else
					residuals[i] = T(0.1)*p[i];
			}
			else if (i >= 66) //fingers
				residuals[i] = T(1.0)*p[i];
			else
				residuals[i] = T(1.0)*p[i];;// *p[i];	//Do nothing*/
		}
		return true;
	}
	const double num_parameters_;
};


class SMPLPosePrior_withDiff : public ceres::CostFunction {
public:
    SMPLPosePrior_withDiff(int num_parameters)
        : num_parameters_(num_parameters)
    {
        assert(num_parameters == SMPLModel::NUM_POSE_PARAMETERS); // for adam
        CostFunction::set_num_residuals(num_parameters_);
        CostFunction::mutable_parameter_block_sizes()->clear();
        CostFunction::mutable_parameter_block_sizes()->push_back(num_parameters_);
        weight[0] = weight[1] = weight[2] = 0.0;

        for (int i = 3; i < 52*3; i++) {
            weight[i] = 0; //Init all weights as 1
        }

		//Regulize 3,6,9 x
        //Regulize some joints
        // weight[3 * 3 + 0] = 1e2;
        // weight[3 * 3 + 1] = 1e2;
        // weight[3 * 3 + 2] = 1e2; //Left foot

        // weight[6 * 3 + 0] = 1e2;
        // weight[6 * 3 + 1] = 1e2;
        // weight[6 * 3 + 2] = 1e2; //Right foot

        // weight[9 * 3 + 0] = 1e2;
        // weight[9 * 3 + 1] = 1e2;
        // weight[9 * 3 + 2] = 1e2; //Left hand

        //weight[23 * 3 + 0] = 1e2;
		//weight[23 * 3 + 1] = 1e2;
        //weight[23 * 3 + 2] = 1e2; //right hand
    }
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        double const* p = parameters[0];
        for (int i = 0; i < num_parameters_; i++) {
            residuals[i] = p[i] * weight[i];
        }

        if (jacobians) {
            if (jacobians[0]) {
                double* jac = jacobians[0];
                std::fill(jac, jac + num_parameters_ * num_parameters_, 0);
                for (int i = 0; i < num_parameters_; i++) {
                    double* jac_row = jac + i * num_parameters_;
                    jac_row[i] = weight[i];
                }
            }
        }
        return true;
    }
    std::array<float, SMPLModel::NUM_POSE_PARAMETERS> weight;

private:
    const int num_parameters_;
};