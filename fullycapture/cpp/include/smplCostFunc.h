#ifndef SMPL_FAST_COST
#define SMPL_FAST_COST
#include "smplFK.h"
#include "smplModel.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include "mainFit.h"
class SMPLFullCost : public ceres::CostFunction {
public:
    SMPLFullCost(const SMPLFitData& fit_data,const SMPLFitOptions &fit_opt)
        : fit_data_(fit_data)
        , fit_opt_(fit_opt)
    {
        SetupCost();

        // setup parent indexes, for fast LBS jacobian computation
        parentIndices[0].clear();
        parentIndices[0].push_back(0);
        for (auto i = 0u; i < SMPLModel::NUM_JOINTS; i++) {
            parentIndices[i] = std::vector<int>(1, i);
            while (parentIndices[i].back() != 0)
                parentIndices[i].emplace_back(fit_data_.smplModel.parent_[parentIndices[i].back()]);
            std::sort(parentIndices[i].begin(), parentIndices[i].end());
        }

        //Vertex Jacobian
        dVdP_data = new double[3 * total_vertex.size() * SMPLModel::NUM_POSE_PARAMETERS];
        dVdc_data = new double[3 * total_vertex.size() * SMPLModel::NUM_SHAPE_COEFFICIENTS];

        //Final Objective Jacobian
        dOdP_data = new double[3 * m_nSmpl2Jtr * SMPLModel::NUM_POSE_PARAMETERS];
        dOdc_data = new double[3 * m_nSmpl2Jtr * SMPLModel::NUM_SHAPE_COEFFICIENTS];
    }
    ~SMPLFullCost()
    {
        delete[] dVdc_data;
        delete[] dVdP_data;
        delete[] dOdP_data;
        delete[] dOdc_data;
    }

    void SetupCost()
    {
        using namespace cv;
        using namespace Eigen;
        //TODO only enable reg_type==TOTAL_JOINTS currently. fix it later.
        //TODO change hard coded reg_type==0 to enum numbers

        assert(fit_opt_.reg_type==0);
        m_nSmpl2Jtr = SMPLModel::NUM_TOTAL_JOINTS;

        int count_vertex = 0;
        for (int k = 0; k < fit_data_.smplModel.J_reg_total_.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(fit_data_.smplModel.J_reg_total_, k); it; ++it) {
                total_vertex.push_back(k);
                count_vertex++;
                break; // now this vertex is used, go to next vertex
            }
        }
        //only use vertices that count in the joint regressor.
        
        //used for differeent representation between regressor and target joints.

        map_regressor_to_constraint.clear();
        for (int i = 0; i < SMPLModel::NUM_TOTAL_JOINTS; i++)
            map_regressor_to_constraint[i] = i;

        //the fitting target, x,y,z,w * N
        m_targetPts.resize(m_nSmpl2Jtr*4);
        m_targetPts.setZero();

        m_targetPts_weight.resize(m_nSmpl2Jtr);

        //buffer is for in-place 
        m_targetPts_weight_buffer.resize(m_nSmpl2Jtr);


        // copy the fitting target in place
        SetupWeight();
        UpdateTarget();

        // start counting from PAF
        res_dim = 3; //x,y,z
        m_nResiduals =  m_nSmpl2Jtr * 3;
        CostFunction::set_num_residuals(m_nResiduals);
        auto parameter_block_sizes = CostFunction::mutable_parameter_block_sizes();
        parameter_block_sizes->clear();
        parameter_block_sizes->push_back(3); // Translation
        parameter_block_sizes->push_back(SMPLModel::NUM_POSE_PARAMETERS); // SMPL Pose
        parameter_block_sizes->push_back(SMPLModel::NUM_SHAPE_COEFFICIENTS); // SMPL Theta
    }

    void SetupWeight()
    {
        //double BODYJOINT_WEIGHT = 1;
        //double HANDJOINT_WEIGHT = 0.75;
        //body joints end in 25
        //hand joints from 20 - 65

        //TODO fix the hard code
        for (int i = 0; i < 25; i++) {
            m_targetPts_weight[i] = fit_opt_.body_joints_weight;
            m_targetPts_weight_buffer[i] = fit_opt_.body_joints_weight;
        }

        for (int i= 25;i< 65;i++){
            m_targetPts_weight[i] = fit_opt_.hand_joints_weight;
            m_targetPts_weight_buffer[i] = fit_opt_.hand_joints_weight;
        }
    }
    void UpdateTarget()
    {
        for (int i = 0; i < m_nSmpl2Jtr; i++){
            m_targetPts.block(4 * i, 0, 4, 1) = fit_data_.totalJoints.col(i);
        }
    }

    //torso, limb, palm, fingers. notice that the torso is always enabled
    void toggle_activate(bool limb,bool palm, bool fingers)
    {
        //fit torso first
        //the torso part.
        for (int ic = 0; ic < 25; ic++) {
            if (ic != 2 && ic != 5 && ic!=8 && ic != 9 && ic != 12) {
                m_targetPts_weight[ic] = double(limb) * m_targetPts_weight_buffer[ic];
            }
        }

        //right hand
        for (int ic=25; ic<45;ic++){
            if (ic != 25+0 && ic != 25+4 && ic!=25+8 && ic != 25+13 && ic != 25+17) {
                m_targetPts_weight[ic] = double(fingers) * m_targetPts_weight_buffer[ic];
            }else{
                m_targetPts_weight[ic] = double(palm) * m_targetPts_weight_buffer[ic];
            }
        }

        //left hand
        for (int ic=45; ic<65;ic++){
            if (ic != 45+0 && ic != 45+4 && ic!=45+8 && ic != 45+13 && ic != 45+17) {
                m_targetPts_weight[ic] = double(fingers) * m_targetPts_weight_buffer[ic];
            }else{
                m_targetPts_weight[ic] = double(palm) * m_targetPts_weight_buffer[ic];
            }
        }
    }

    void toggle_rigid_body(bool rigid)
    {
        rigid_body = rigid;
    }

    void select_lbs(
        const double* c,
        const Eigen::VectorXd& T, // transformation
        const MatrixXdr& dTdP,
        const MatrixXdr& dTdc,
        MatrixXdr& outVert,
        double* dVdP_data, //output
        double* dVdc_data //output
        ) const;

    void select_lbs(
        const double* c,
        const Eigen::VectorXd& T, // transformation
        MatrixXdr& outVert) const;

    void SparseRegress(const Eigen::SparseMatrix<double>& reg, const double* V_data, const double* dVdP_data, const double* dVdc_data,
        double* J_data, double* dJdP_data, double* dJdc_data) const;

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    //members
    Eigen::VectorXd m_targetPts_weight; //residual weight
    Eigen::VectorXd m_targetPts_weight_buffer;
    int m_nSmpl2Jtr; //Smpl fit to Joints
    int m_nResiduals;
    int res_dim; // number of residuals per joint / vertex constraints

private:
    // input data
    const SMPLFitData& fit_data_;
    const SMPLFitOptions& fit_opt_;

    // setting
    bool rigid_body;
    // data for joint / projection fitting
    Eigen::VectorXd m_targetPts;
    // data for vertex fitting
    std::vector<int> total_vertex; // all vertices that needs to be computed

    // data for PAF fitting
    static const int DEFINED_INNER_CONSTRAINTS = 1;
    
    // parent index
    std::array<std::vector<int>, SMPLModel::NUM_JOINTS> parentIndices;
    // jacobians
    double* dVdP_data;
    double* dVdc_data;
    double* dOdP_data;
    double* dOdc_data;

    std::map<int, int> map_regressor_to_constraint;
};

#endif