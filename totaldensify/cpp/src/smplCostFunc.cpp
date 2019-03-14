#include"smplCostFunc.h"
#include"smplModel.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <omp.h>
void ComputedTrdc_SMPL(const double* dTrdJ_data, const double* dJdc_data, double* dTrdc_data, const std::array<std::vector<int>, SMPLModel::NUM_JOINTS>& parentIndices)
{
    // const Eigen::Map<const Eigen::Matrix<double, 5 * 3 * SMPLModel::NUM_JOINTS, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor>> dTrdJ(dTrdJ_data);
    // const Eigen::Map<const Eigen::Matrix<double, 3 * SMPLModel::NUM_JOINTS, SMPLModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor>> dJdc(dJdc_data);
    // Eigen::Map<Eigen::Matrix<double, 5 * 3 * SMPLModel::NUM_JOINTS, SMPLModel::NUM_SHAPE_aCOEFFICIENTS, Eigen::RowMajor>> dTrdc(dTrdc_data);
    const int ncol = 3 * SMPLModel::NUM_JOINTS;
    const int ncol_out = SMPLModel::NUM_SHAPE_COEFFICIENTS;
    std::fill(dTrdc_data, dTrdc_data + 5 * 3 * SMPLModel::NUM_JOINTS * SMPLModel::NUM_SHAPE_COEFFICIENTS, 0);
    for (int i = 0; i < SMPLModel::NUM_JOINTS; i++) {
        // 12 rows to take care of
        // only row 12 * i + 4 * j + 3 is non-zero
        for (int j = 0; j < 3; j++) {
            const auto* dTrdJ_row = dTrdJ_data + (12 * i + 4 * j + 3) * ncol;
            auto* dTrdc_row = dTrdc_data + (12 * i + 4 * j + 3) * ncol_out;
            for (auto& ipar : parentIndices[i]) {
                const auto ipar3 = 3 * ipar;
                for (int c = 0; c < ncol_out; c++) {
                    dTrdc_row[c] += dTrdJ_row[ipar3] * dJdc_data[ipar3 * ncol_out + c]
                        + dTrdJ_row[ipar3 + 1] * dJdc_data[(ipar3 + 1) * ncol_out + c]
                        + dTrdJ_row[ipar3 + 2] * dJdc_data[(ipar3 + 2) * ncol_out + c];
                }
            }
        }
    }

    for (int i = 0; i < SMPLModel::NUM_JOINTS; i++) {
        // 3 rows to take care of
        // only row 12 * i + 4 * j + 3 is non-zero
        for (int j = 0; j < 3; j++) {
            const auto* dTrdJ_row = dTrdJ_data + 4 * 3 * SMPLModel::NUM_JOINTS * 3 * SMPLModel::NUM_JOINTS + (3 * i + j) * ncol;
            auto* dTrdc_row = dTrdc_data + 4 * 3 * SMPLModel::NUM_JOINTS * SMPLModel::NUM_SHAPE_COEFFICIENTS + (3 * i + j) * ncol_out;
            for (auto& ipar : parentIndices[i]) {
                const auto ipar3 = 3 * ipar;
                for (int c = 0; c < ncol_out; c++) {
                    dTrdc_row[c] += dTrdJ_row[ipar3] * dJdc_data[ipar3 * ncol_out + c]
                        + dTrdJ_row[ipar3 + 1] * dJdc_data[(ipar3 + 1) * ncol_out + c]
                        + dTrdJ_row[ipar3 + 2] * dJdc_data[(ipar3 + 2) * ncol_out + c];
                }
            }
        }
    }
}

const int SMPLFullCost::DEFINED_INNER_CONSTRAINTS;

bool SMPLFullCost::Evaluate(double const* const* parameters,
    double* residuals,
    double** jacobians) const
{
    // const auto start_iter = std::chrono::high_resolution_clock::now();
    using namespace Eigen;
    typedef double T;

    const T* t = parameters[0]; //Trans
    const T* p = parameters[1]; //pose
    const T* c = parameters[2]; //betas

    Map<const Vector3d> t_vec(t);
    Map<const Matrix<double, Dynamic, 1>> c_bodyshape(c, SMPLModel::NUM_SHAPE_COEFFICIENTS);

    // 0st step: Compute all the current joints
    Matrix<double, SMPLModel::NUM_JOINTS, 3, RowMajor> J;
    Map<Matrix<double, Dynamic, 1>> J_vec(J.data(), SMPLModel::NUM_JOINTS * 3);
    //'

    J_vec = fit_data_.smplModel.J_mu_ + fit_data_.smplModel.dJdc_ * c_bodyshape;
    //J_vec = J_reg * (mu_+ U_*c) = J_mu + dJdc*c
    // 1st step: forward kinematics, pose_2_transforms
    const int J_trans = (SMPLModel::NUM_JOINTS)*3 * 5; // transform 3 * 4 + joint location 3 * 1

    Matrix<double, Dynamic, 3 * SMPLModel::NUM_JOINTS, RowMajor> dTrdP(J_trans, 3 * SMPLModel::NUM_JOINTS);
    Matrix<double, Dynamic, 3 * SMPLModel::NUM_JOINTS, RowMajor> dTrdJ(J_trans, 3 * SMPLModel::NUM_JOINTS);

    VectorXd transforms_joint(3 * SMPLModel::NUM_JOINTS * 4 + 3 * SMPLModel::NUM_JOINTS);
    const double* p2t_parameters[2] = { p, J.data() };
    double* p2t_residuals = transforms_joint.data();
    double* p2t_jacobians[2] = { dTrdP.data(), dTrdJ.data() };

    PoseToTransform_SMPLFull_withDiff p2t(fit_data_.smplModel, parentIndices);

    // const auto start_FK = std::chrono::high_resolution_clock::now();
    p2t.Evaluate(p2t_parameters, p2t_residuals, jacobians ? p2t_jacobians : nullptr);

    // const auto start_transJ = std::chrono::high_resolution_clock::now();
    // MatrixXdr dTJdP = dTrdP.block(3 * SMPLModel::NUM_JOINTS * 4, 0, 3 * SMPLModel::NUM_JOINTS, 3 * SMPLModel::NUM_JOINTS);
    Map<MatrixXdr> dTJdP(dTrdP.data() + 3 * SMPLModel::NUM_JOINTS * 4 * 3 * SMPLModel::NUM_JOINTS, 3 * SMPLModel::NUM_JOINTS, 3 * SMPLModel::NUM_JOINTS);
    // The following lines are equiv to MatrixXdr dTrdc = dTrdJ * fit_data_.adam.dJdc_;
    MatrixXdr dTrdc(J_trans, SMPLModel::NUM_SHAPE_COEFFICIENTS);
    if (jacobians)
        ComputedTrdc_SMPL(dTrdJ.data(), fit_data_.smplModel.dJdc_.data(), dTrdc.data(), parentIndices);
    // MatrixXdr dTJdc = dTrdc.block(3 * SMPLModel::NUM_JOINTS * 4, 0, 3 * SMPLModel::NUM_JOINTS, SMPLModel::NUM_SHAPE_COEFFICIENTS);
    Map<MatrixXdr> dTJdc(dTrdc.data() + 3 * SMPLModel::NUM_JOINTS * 4 * SMPLModel::NUM_SHAPE_COEFFICIENTS, 3 * SMPLModel::NUM_JOINTS, SMPLModel::NUM_SHAPE_COEFFICIENTS);
    VectorXd outJoint = transforms_joint.block(3 * SMPLModel::NUM_JOINTS * 4, 0, 3 * SMPLModel::NUM_JOINTS, 1); // outJoint
    for (auto i = 0u; i < SMPLModel::NUM_JOINTS; i++)
        outJoint.block(3 * i, 0, 3, 1) += t_vec;
    // const auto duration_transJ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_transJ).count();

    MatrixXdr outVert(total_vertex.size(), 3);
    Map<MatrixXdr> dVdP(dVdP_data, 3 * total_vertex.size(), SMPLModel::NUM_POSE_PARAMETERS);
    Map<MatrixXdr> dVdc(dVdc_data, 3 * total_vertex.size(), SMPLModel::NUM_SHAPE_COEFFICIENTS);

    // const auto start_LBS = std::chrono::high_resolution_clock::now();
    if (jacobians)
        select_lbs(c, transforms_joint, dTrdP, dTrdc, outVert, dVdP_data, dVdc_data);
    else
        select_lbs(c, transforms_joint, outVert);
    // const auto duration_LBS = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_LBS).count();
    outVert.rowwise() += t_vec.transpose();
    std::array<double*, 3> out_data{ { outJoint.data(), outVert.data(), nullptr } };
    std::array<Map<MatrixXdr>*, 3> dodP = { { &dTJdP, &dVdP, nullptr } }; // array of reference is not allowed, only array of pointer
    std::array<Map<MatrixXdr>*, 3> dodc = { { &dTJdc, &dVdc, nullptr } };
    // 2nd step: compute the target joints (copy from FK)
    // const auto start_target = std::chrono::high_resolution_clock::now();
    // Arrange the Output Joints & Vertex to the order of constraints
    VectorXd tempJoints(3 * m_nSmpl2Jtr);
    Map<MatrixXdr> dOdP(dOdP_data, 3 * m_nSmpl2Jtr, SMPLModel::NUM_POSE_PARAMETERS);
    Map<MatrixXdr> dOdc(dOdc_data, 3 * m_nSmpl2Jtr, SMPLModel::NUM_SHAPE_COEFFICIENTS);
    Map<const Matrix<double, Dynamic, Dynamic, RowMajor>> pose_param(p, SMPLModel::NUM_JOINTS, 3);

    //TODO support multiple regressor
    assert(fit_opt_.reg_type == 2);
    if (jacobians)
        SparseRegress(fit_data_.smplModel.J_reg_total_, outVert.data(), dVdP_data, dVdc_data, tempJoints.data(), dOdP.data(), dOdc.data());
    else 
        SparseRegress(fit_data_.smplModel.J_reg_total_, outVert.data(), nullptr, nullptr, tempJoints.data(), nullptr, nullptr);

    //std::cout << tempJoints << std::endl;

    out_data[2] = tempJoints.data();
    if (jacobians) {
        dodP[2] = &dOdP;
        dodc[2] = &dOdc;
    }

    // 3rd step: set residuals
    Map<VectorXd> res(residuals, m_nResiduals);
    const auto* targetPts = m_targetPts.data();
    const auto* targetPtsWeight = m_targetPts_weight.data();

    for (int i = 0; i < m_nSmpl2Jtr; i++) {
        if (targetPts[4 * i + 3] < 1e-3)
            res.block(res_dim * i, 0, 3, 1).setZero();
        else
            res.block(res_dim * i, 0, 3, 1) = m_targetPts_weight[i] * (tempJoints.block(3 * i, 0, 3, 1) - m_targetPts.block(4 * i, 0, 3, 1));
    }

    // 4th step: set jacobians
    if (jacobians) {
        if (jacobians[0]) {
            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> drdt(jacobians[0], m_nResiduals, 3);
            for (int i = 0; i < m_nSmpl2Jtr; i++) {
                if (targetPts[4 * i+3]<1e-3)
                    drdt.block(res_dim * i, 0, 3, 3).setZero();
                else
                    drdt.block(res_dim * i, 0, 3, 3) = m_targetPts_weight[i] * Matrix<double, 3, 3>::Identity();
            }
        }
        if (jacobians[1]) {
            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> dr_dPose(jacobians[1], m_nResiduals, SMPLModel::NUM_POSE_PARAMETERS);

            for (int i = 0; i < m_nSmpl2Jtr; i++) {
                if (targetPts[4 * i+3]<1e-3) {
                    std::fill(dr_dPose.data() + res_dim * i * SMPLModel::NUM_POSE_PARAMETERS,
                        dr_dPose.data() + (3 + res_dim * i) * SMPLModel::NUM_POSE_PARAMETERS, 0);
                    // dr_dPose.block(res_dim * (i + offset), 0, 3, SMPLModel::NUM_POSE_PARAMETERS).setZero();
                } else
                    dr_dPose.block(res_dim * i, 0, 3, SMPLModel::NUM_POSE_PARAMETERS) = m_targetPts_weight[i] * dOdP.block(3 * i, 0, 3, SMPLModel::NUM_POSE_PARAMETERS);
            }
            if (rigid_body)
                dr_dPose.block(0, 3, m_nResiduals, SMPLModel::NUM_POSE_PARAMETERS - 3).setZero();
        }
        if (rigid_body) {
            if (jacobians[2]) {
                std::fill(jacobians[2], jacobians[2] + m_nResiduals * SMPLModel::NUM_SHAPE_COEFFICIENTS, 0);
            }
            return true;
        }

        if (jacobians[2]) {
            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> dr_dCoeff(jacobians[2], m_nResiduals, SMPLModel::NUM_SHAPE_COEFFICIENTS);
            for (int i = 0; i < m_nSmpl2Jtr; i++) {
                if (targetPts[4 * i+3]<1e-3) {
                    std::fill(dr_dCoeff.data() + res_dim * i * SMPLModel::NUM_SHAPE_COEFFICIENTS,
                        dr_dCoeff.data() + (3 + res_dim * i) * SMPLModel::NUM_SHAPE_COEFFICIENTS, 0);
                } else
                    dr_dCoeff.block(res_dim * i, 0, 3, SMPLModel::NUM_SHAPE_COEFFICIENTS) = m_targetPts_weight[i] * dOdc.block(3 * i, 0, 3, SMPLModel::NUM_SHAPE_COEFFICIENTS);
            }

        }
    }
    return true;
}

// LBS with Jacobian
void SMPLFullCost::select_lbs(
    const double* c,
    const Eigen::VectorXd& T, // transformation
    const MatrixXdr& dTdP,
    const MatrixXdr& dTdc,
    MatrixXdr& outVert,
    double* dVdP_data, //output
    double* dVdc_data //output
    ) const
{
    // read adam model and corres_vertex2targetpt from the class member
    using namespace Eigen;
    assert(outVert.rows() == total_vertex.size());
    std::fill(dVdc_data, dVdc_data + 3 * total_vertex.size() * SMPLModel::NUM_SHAPE_COEFFICIENTS, 0); // dVdc.setZero();
    std::fill(dVdP_data, dVdP_data + 3 * total_vertex.size() * SMPLModel::NUM_POSE_PARAMETERS, 0); // dVdP.setZero();
    const double* dTdc_data = dTdc.data();
    const double* dTdP_data = dTdP.data();
    const double* dV0dc_data = fit_data_.smplModel.U_.data(); //shape basis
    const double* meanshape_data = fit_data_.smplModel.mu_.data();

    for (auto i = 0u; i < total_vertex.size(); i++) {
        const int idv = total_vertex[i];
        // compute the default vertex, v0 is a column vector
        // The following lines are equivalent to
        // MatrixXd v0 = fit_data_.adam.m_meanshape.block(3 * idv, 0, 3, 1) + fit_data_.adam.m_shapespace_u.block(3 * idv, 0, 3, SMPLModel::NUM_SHAPE_COEFFICIENTS) * c_bodyshape;
        MatrixXd v0(3, 1);
        auto* v0_data = v0.data();
        v0_data[0] = meanshape_data[3 * idv + 0];
        v0_data[1] = meanshape_data[3 * idv + 1];
        v0_data[2] = meanshape_data[3 * idv + 2];
        const int nrow = fit_data_.smplModel.U_.rows();
        for (int ic = 0; ic < SMPLModel::NUM_SHAPE_COEFFICIENTS; ic++) {
            v0_data[0] += dV0dc_data[ic * nrow + 3 * idv + 0] * c[ic];
            v0_data[1] += dV0dc_data[ic * nrow + 3 * idv + 1] * c[ic];
            v0_data[2] += dV0dc_data[ic * nrow + 3 * idv + 2] * c[ic];
        }
        auto* outVrow_data = outVert.data() + 3 * i;
        outVrow_data[0] = outVrow_data[1] = outVrow_data[2] = 0;
        for (int idj = 0; idj < SMPLModel::NUM_JOINTS; idj++) {
            const double w = fit_data_.smplModel.W_(idv, idj);
            if (w) {
                const auto* const Trow_data = T.data() + 12 * idj;
                outVrow_data[0] += w * (Trow_data[0] * v0_data[0] + Trow_data[1] * v0_data[1] + Trow_data[2] * v0_data[2] + Trow_data[3]);
                outVrow_data[1] += w * (Trow_data[4] * v0_data[0] + Trow_data[5] * v0_data[1] + Trow_data[6] * v0_data[2] + Trow_data[7]);
                outVrow_data[2] += w * (Trow_data[8] * v0_data[0] + Trow_data[9] * v0_data[1] + Trow_data[10] * v0_data[2] + Trow_data[11]);

                const int ncol = SMPLModel::NUM_POSE_PARAMETERS;
                double* dVdP_row0 = dVdP_data + (i * 3) * SMPLModel::NUM_POSE_PARAMETERS;
                double* dVdP_row1 = dVdP_data + (i * 3 + 1) * SMPLModel::NUM_POSE_PARAMETERS;
                double* dVdP_row2 = dVdP_data + (i * 3 + 2) * SMPLModel::NUM_POSE_PARAMETERS;
                const double* dTdP_base = dTdP_data + idj * 12 * SMPLModel::NUM_POSE_PARAMETERS;
                for (int j = 0; j < parentIndices[idj].size(); j++) {
                    const int idp = parentIndices[idj][j];
                    // The following lines are equiv to
                    // dVdP(i * 3 + 0, 3 * idp + 0) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 0 * 4 + 0, 3 * idp + 0) + v0_data[1] * dTdP(idj * 3 * 4 + 0 * 4 + 1, 3 * idp + 0) + v0_data[2] * dTdP(idj * 3 * 4 + 0 * 4 + 2, 3 * idp + 0) + dTdP(idj * 12 + 0 * 4 + 3, 3 * idp + 0));
                    // dVdP(i * 3 + 1, 3 * idp + 0) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 1 * 4 + 0, 3 * idp + 0) + v0_data[1] * dTdP(idj * 3 * 4 + 1 * 4 + 1, 3 * idp + 0) + v0_data[2] * dTdP(idj * 3 * 4 + 1 * 4 + 2, 3 * idp + 0) + dTdP(idj * 12 + 1 * 4 + 3, 3 * idp + 0));
                    // dVdP(i * 3 + 2, 3 * idp + 0) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 2 * 4 + 0, 3 * idp + 0) + v0_data[1] * dTdP(idj * 3 * 4 + 2 * 4 + 1, 3 * idp + 0) + v0_data[2] * dTdP(idj * 3 * 4 + 2 * 4 + 2, 3 * idp + 0) + dTdP(idj * 12 + 2 * 4 + 3, 3 * idp + 0));
                    // dVdP(i * 3 + 0, 3 * idp + 1) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 0 * 4 + 0, 3 * idp + 1) + v0_data[1] * dTdP(idj * 3 * 4 + 0 * 4 + 1, 3 * idp + 1) + v0_data[2] * dTdP(idj * 3 * 4 + 0 * 4 + 2, 3 * idp + 1) + dTdP(idj * 12 + 0 * 4 + 3, 3 * idp + 1));
                    // dVdP(i * 3 + 1, 3 * idp + 1) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 1 * 4 + 0, 3 * idp + 1) + v0_data[1] * dTdP(idj * 3 * 4 + 1 * 4 + 1, 3 * idp + 1) + v0_data[2] * dTdP(idj * 3 * 4 + 1 * 4 + 2, 3 * idp + 1) + dTdP(idj * 12 + 1 * 4 + 3, 3 * idp + 1));
                    // dVdP(i * 3 + 2, 3 * idp + 1) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 2 * 4 + 0, 3 * idp + 1) + v0_data[1] * dTdP(idj * 3 * 4 + 2 * 4 + 1, 3 * idp + 1) + v0_data[2] * dTdP(idj * 3 * 4 + 2 * 4 + 2, 3 * idp + 1) + dTdP(idj * 12 + 2 * 4 + 3, 3 * idp + 1));
                    // dVdP(i * 3 + 0, 3 * idp + 2) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 0 * 4 + 0, 3 * idp + 2) + v0_data[1] * dTdP(idj * 3 * 4 + 0 * 4 + 1, 3 * idp + 2) + v0_data[2] * dTdP(idj * 3 * 4 + 0 * 4 + 2, 3 * idp + 2) + dTdP(idj * 12 + 0 * 4 + 3, 3 * idp + 2));
                    // dVdP(i * 3 + 1, 3 * idp + 2) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 1 * 4 + 0, 3 * idp + 2) + v0_data[1] * dTdP(idj * 3 * 4 + 1 * 4 + 1, 3 * idp + 2) + v0_data[2] * dTdP(idj * 3 * 4 + 1 * 4 + 2, 3 * idp + 2) + dTdP(idj * 12 + 1 * 4 + 3, 3 * idp + 2));
                    // dVdP(i * 3 + 2, 3 * idp + 2) +=
                    //     w * (v0_data[0] * dTdP(idj * 3 * 4 + 2 * 4 + 0, 3 * idp + 2) + v0_data[1] * dTdP(idj * 3 * 4 + 2 * 4 + 1, 3 * idp + 2) + v0_data[2] * dTdP(idj * 3 * 4 + 2 * 4 + 2, 3 * idp + 2) + dTdP(idj * 12 + 2 * 4 + 3, 3 * idp + 2));
                    dVdP_row0[3 * idp + 0] += w * (v0_data[0] * dTdP_base[(0 * 4 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dTdP_base[(0 * 4 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dTdP_base[(0 * 4 + 2) * ncol + 3 * idp + 0] + dTdP_base[(0 * 4 + 3) * ncol + 3 * idp + 0]);
                    dVdP_row1[3 * idp + 0] += w * (v0_data[0] * dTdP_base[(1 * 4 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dTdP_base[(1 * 4 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dTdP_base[(1 * 4 + 2) * ncol + 3 * idp + 0] + dTdP_base[(1 * 4 + 3) * ncol + 3 * idp + 0]);
                    dVdP_row2[3 * idp + 0] += w * (v0_data[0] * dTdP_base[(2 * 4 + 0) * ncol + 3 * idp + 0] + v0_data[1] * dTdP_base[(2 * 4 + 1) * ncol + 3 * idp + 0] + v0_data[2] * dTdP_base[(2 * 4 + 2) * ncol + 3 * idp + 0] + dTdP_base[(2 * 4 + 3) * ncol + 3 * idp + 0]);
                    dVdP_row0[3 * idp + 1] += w * (v0_data[0] * dTdP_base[(0 * 4 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dTdP_base[(0 * 4 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dTdP_base[(0 * 4 + 2) * ncol + 3 * idp + 1] + dTdP_base[(0 * 4 + 3) * ncol + 3 * idp + 1]);
                    dVdP_row1[3 * idp + 1] += w * (v0_data[0] * dTdP_base[(1 * 4 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dTdP_base[(1 * 4 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dTdP_base[(1 * 4 + 2) * ncol + 3 * idp + 1] + dTdP_base[(1 * 4 + 3) * ncol + 3 * idp + 1]);
                    dVdP_row2[3 * idp + 1] += w * (v0_data[0] * dTdP_base[(2 * 4 + 0) * ncol + 3 * idp + 1] + v0_data[1] * dTdP_base[(2 * 4 + 1) * ncol + 3 * idp + 1] + v0_data[2] * dTdP_base[(2 * 4 + 2) * ncol + 3 * idp + 1] + dTdP_base[(2 * 4 + 3) * ncol + 3 * idp + 1]);
                    dVdP_row0[3 * idp + 2] += w * (v0_data[0] * dTdP_base[(0 * 4 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dTdP_base[(0 * 4 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dTdP_base[(0 * 4 + 2) * ncol + 3 * idp + 2] + dTdP_base[(0 * 4 + 3) * ncol + 3 * idp + 2]);
                    dVdP_row1[3 * idp + 2] += w * (v0_data[0] * dTdP_base[(1 * 4 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dTdP_base[(1 * 4 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dTdP_base[(1 * 4 + 2) * ncol + 3 * idp + 2] + dTdP_base[(1 * 4 + 3) * ncol + 3 * idp + 2]);
                    dVdP_row2[3 * idp + 2] += w * (v0_data[0] * dTdP_base[(2 * 4 + 0) * ncol + 3 * idp + 2] + v0_data[1] * dTdP_base[(2 * 4 + 1) * ncol + 3 * idp + 2] + v0_data[2] * dTdP_base[(2 * 4 + 2) * ncol + 3 * idp + 2] + dTdP_base[(2 * 4 + 3) * ncol + 3 * idp + 2]);
                }

                // Note that dV0dc is column major
                const int ncolc = SMPLModel::NUM_SHAPE_COEFFICIENTS;
                double* dVdc_row0 = dVdc_data + (i * 3 + 0) * ncolc;
                double* dVdc_row1 = dVdc_data + (i * 3 + 1) * ncolc;
                double* dVdc_row2 = dVdc_data + (i * 3 + 2) * ncolc;
                const double* dTdc_row0 = dTdc_data + (idj * 12 + 0 * 4 + 3) * ncolc;
                const double* dTdc_row1 = dTdc_data + (idj * 12 + 1 * 4 + 3) * ncolc;
                const double* dTdc_row2 = dTdc_data + (idj * 12 + 2 * 4 + 3) * ncolc;
                for (int idc = 0; idc < SMPLModel::NUM_SHAPE_COEFFICIENTS; idc++) {
                    // The following lines are equiv to
                    // dVdc(i * 3 + 0, idc) +=
                    //     w * (dV0dc(idv * 3 + 0, idc) * Trow_data[0 * 4 + 0] + dV0dc(idv * 3 + 1, idc) * Trow_data[0 * 4 + 1] + dV0dc(idv * 3 + 2, idc) * Trow_data[0 * 4 + 2] + dTdc(idj * 12 + 0 * 4 + 3, idc));
                    // dVdc(i * 3 + 1, idc) +=
                    //     w * (dV0dc(idv * 3 + 0, idc) * Trow_data[1 * 4 + 0] + dV0dc(idv * 3 + 1, idc) * Trow_data[1 * 4 + 1] + dV0dc(idv * 3 + 2, idc) * Trow_data[1 * 4 + 2] + dTdc(idj * 12 + 1 * 4 + 3, idc));
                    // dVdc(i * 3 + 2, idc) +=
                    //     w * (dV0dc(idv * 3 + 0, idc) * Trow_data[2 * 4 + 0] + dV0dc(idv * 3 + 1, idc) * Trow_data[2 * 4 + 1] + dV0dc(idv * 3 + 2, idc) * Trow_data[2 * 4 + 2] + dTdc(idj * 12 + 2 * 4 + 3, idc));
                    dVdc_row0[idc] += w * (dV0dc_data[idc * nrow + idv * 3 + 0] * Trow_data[0 * 4 + 0] + dV0dc_data[idc * nrow + idv * 3 + 1] * Trow_data[0 * 4 + 1] + dV0dc_data[idc * nrow + idv * 3 + 2] * Trow_data[0 * 4 + 2] + dTdc_row0[idc]);
                    dVdc_row1[idc] += w * (dV0dc_data[idc * nrow + idv * 3 + 0] * Trow_data[1 * 4 + 0] + dV0dc_data[idc * nrow + idv * 3 + 1] * Trow_data[1 * 4 + 1] + dV0dc_data[idc * nrow + idv * 3 + 2] * Trow_data[1 * 4 + 2] + dTdc_row1[idc]);
                    dVdc_row2[idc] += w * (dV0dc_data[idc * nrow + idv * 3 + 0] * Trow_data[2 * 4 + 0] + dV0dc_data[idc * nrow + idv * 3 + 1] * Trow_data[2 * 4 + 1] + dV0dc_data[idc * nrow + idv * 3 + 2] * Trow_data[2 * 4 + 2] + dTdc_row2[idc]);
                }
            }
        }
    }
}


// LBS w/o jacobian
void SMPLFullCost::select_lbs(
    const double* c,
    const Eigen::VectorXd& T, // transformation
    MatrixXdr& outVert) const
{
    // read adam model and total_vertex from the class member
    using namespace Eigen;
    // Map< const Matrix<double, Dynamic, 1> > c_bodyshape(c, SMPLModel::NUM_SHAPE_COEFFICIENTS);
    assert(outVert.rows() == total_vertex.size());
    // const Eigen::MatrixXd& dV0dc = fit_data_.adam.m_shapespace_u;
    const double* dV0dc_data = fit_data_.smplModel.U_.data();
    const double* meanshape_data = fit_data_.smplModel.mu_.data();

    for (auto i = 0u; i < total_vertex.size(); i++) {
        const int idv = total_vertex[i];
        // compute the default vertex, v0 is a column vector
        // The following lines are equivalent to
        // MatrixXd v0 = fit_data_.adam.m_meanshape.block(3 * idv, 0, 3, 1) + fit_data_.adam.m_shapespace_u.block(3 * idv, 0, 3, SMPLModel::NUM_SHAPE_COEFFICIENTS) * c_bodyshape;
        MatrixXd v0(3, 1);
        auto* v0_data = v0.data();
        v0_data[0] = meanshape_data[3 * idv + 0];
        v0_data[1] = meanshape_data[3 * idv + 1];
        v0_data[2] = meanshape_data[3 * idv + 2];
        const int nrow = fit_data_.smplModel.U_.rows();
        for (int ic = 0; ic < SMPLModel::NUM_SHAPE_COEFFICIENTS; ic++) {
            v0_data[0] += dV0dc_data[ic * nrow + 3 * idv + 0] * c[ic];
            v0_data[1] += dV0dc_data[ic * nrow + 3 * idv + 1] * c[ic];
            v0_data[2] += dV0dc_data[ic * nrow + 3 * idv + 2] * c[ic];
        }
        auto* outVrow_data = outVert.data() + 3 * i;
        outVrow_data[0] = outVrow_data[1] = outVrow_data[2] = 0;
        for (int idj = 0; idj < SMPLModel::NUM_JOINTS; idj++) {
            const double w = fit_data_.smplModel.W_(idv, idj);
            if (w) {
                const auto* const Trow_data = T.data() + 12 * idj;
                outVrow_data[0] += w * (Trow_data[0] * v0_data[0] + Trow_data[1] * v0_data[1] + Trow_data[2] * v0_data[2] + Trow_data[3]);
                outVrow_data[1] += w * (Trow_data[4] * v0_data[0] + Trow_data[5] * v0_data[1] + Trow_data[6] * v0_data[2] + Trow_data[7]);
                outVrow_data[2] += w * (Trow_data[8] * v0_data[0] + Trow_data[9] * v0_data[1] + Trow_data[10] * v0_data[2] + Trow_data[11]);
            }
        }
    }
}

void SMPLFullCost::SparseRegress(const Eigen::SparseMatrix<double>& reg, const double* V_data, const double* dVdP_data, const double* dVdc_data,
    double* J_data, double* dJdP_data, double* dJdc_data) const
{
    const int num_J = m_nSmpl2Jtr;
    std::fill(J_data, J_data + 3 * num_J, 0);
    std::array<bool, SMPLModel::NUM_VERTICES> vertex_covered = {}; // whether this vertex is already covered once in the regressors, init to false.
    for (int ic = 0; ic < total_vertex.size(); ic++) {
        const int c = total_vertex[ic];
        if (vertex_covered[c])
            continue;
        for (Eigen::SparseMatrix<double>::InnerIterator it(reg, c); it; ++it) {
            const int r = it.row();
            auto search = map_regressor_to_constraint.find(r);
            if (search == map_regressor_to_constraint.end())
                continue; // This joint is not used for constraint
            const int ind_constraint = search->second;
            const double value = it.value();
            J_data[3 * ind_constraint + 0] += value * V_data[3 * ic + 0];
            J_data[3 * ind_constraint + 1] += value * V_data[3 * ic + 1];
            J_data[3 * ind_constraint + 2] += value * V_data[3 * ic + 2];
        }
        vertex_covered[c] = true;
    }

    if (dVdP_data != nullptr) // need to pass back the correct Jacobian
    {
        assert(dVdc_data != nullptr && dJdP_data != nullptr && dJdc_data != nullptr);
        std::fill(dJdP_data, dJdP_data + 3 * num_J * SMPLModel::NUM_POSE_PARAMETERS, 0.0);
        std::fill(dJdc_data, dJdc_data + 3 * num_J * SMPLModel::NUM_SHAPE_COEFFICIENTS, 0.0);
        std::fill(vertex_covered.data(), vertex_covered.data() + vertex_covered.size(), false);
        for (int ic = 0; ic < total_vertex.size(); ic++) {
            const int c = total_vertex[ic];
            if (vertex_covered[c])
                continue;
            for (Eigen::SparseMatrix<double>::InnerIterator it(reg, c); it; ++it) {
                const int r = it.row();
                auto search = map_regressor_to_constraint.find(r);
                if (search == map_regressor_to_constraint.end())
                    continue; // This joint is not used for constraint
                const int ind_constraint = search->second;
                const double value = it.value();
                for (int i = 0; i < SMPLModel::NUM_POSE_PARAMETERS; i++) {
                    dJdP_data[(3 * ind_constraint + 0) * SMPLModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 0) * SMPLModel::NUM_POSE_PARAMETERS + i];
                    dJdP_data[(3 * ind_constraint + 1) * SMPLModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 1) * SMPLModel::NUM_POSE_PARAMETERS + i];
                    dJdP_data[(3 * ind_constraint + 2) * SMPLModel::NUM_POSE_PARAMETERS + i] += value * dVdP_data[(3 * ic + 2) * SMPLModel::NUM_POSE_PARAMETERS + i];
                }
                for (int i = 0; i < SMPLModel::NUM_SHAPE_COEFFICIENTS; i++) {
                    dJdc_data[(3 * ind_constraint + 0) * SMPLModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 0) * SMPLModel::NUM_SHAPE_COEFFICIENTS + i];
                    dJdc_data[(3 * ind_constraint + 1) * SMPLModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 1) * SMPLModel::NUM_SHAPE_COEFFICIENTS + i];
                    dJdc_data[(3 * ind_constraint + 2) * SMPLModel::NUM_SHAPE_COEFFICIENTS + i] += value * dVdc_data[(3 * ic + 2) * SMPLModel::NUM_SHAPE_COEFFICIENTS + i];
                }
            }
            vertex_covered[c] = true;
        }
    }
}