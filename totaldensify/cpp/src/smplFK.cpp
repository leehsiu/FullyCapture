#include "smplFK.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <chrono>
#include <iostream>

bool PoseToTransform_SMPLFull_withDiff::Evaluate(double const* const* parameters,
    double* residuals,
    double** jacobians) const
{
    using namespace Eigen;
    int numColumn = SMPLModel::NUM_JOINTS * 3;

    const double* pose = parameters[0];
    const double* joints = parameters[1];
    Eigen::Map<const Eigen::Matrix<double, SMPLModel::NUM_JOINTS, 3, Eigen::RowMajor>> P(pose);
    Eigen::Map<const Matrix<double, SMPLModel::NUM_JOINTS, 3, RowMajor>> J0(joints);
    Eigen::Map<Matrix<double, 3 * SMPLModel::NUM_JOINTS, 4, RowMajor>> outT(residuals);
    Eigen::Map<Matrix<double, SMPLModel::NUM_JOINTS, 3, RowMajor>> outJoint(residuals + 3 * SMPLModel::NUM_JOINTS * 4);

    Map<Matrix<double, 4 * SMPLModel::NUM_JOINTS * 3, SMPLModel::NUM_JOINTS * 3, RowMajor>> dTrdP(jacobians ? jacobians[0] : nullptr);
    Map<Matrix<double, SMPLModel::NUM_JOINTS * 3, SMPLModel::NUM_JOINTS * 3, RowMajor>> dJdP(jacobians ? jacobians[0] + SMPLModel::NUM_JOINTS * SMPLModel::NUM_JOINTS * 36 : nullptr);

    Map<Matrix<double, 4 * SMPLModel::NUM_JOINTS * 3, SMPLModel::NUM_JOINTS * 3, RowMajor>> dTrdJ(jacobians ? jacobians[1] : nullptr);
    Map<Matrix<double, SMPLModel::NUM_JOINTS * 3, SMPLModel::NUM_JOINTS * 3, RowMajor>> dJdJ(jacobians ? jacobians[1] + SMPLModel::NUM_JOINTS * SMPLModel::NUM_JOINTS * 36 : nullptr);
    // fill in dTrdJ first, because it is sparse, only dMtdJ is none-0.
    if (jacobians)
        std::fill(jacobians[1], jacobians[1] + 36 * SMPLModel::NUM_JOINTS * SMPLModel::NUM_JOINTS, 0.0);

    Matrix<double, 3, 3, RowMajor> R; // Interface with ceres
    Matrix<double, 9, 3 * SMPLModel::NUM_JOINTS, RowMajor> dRdP(9, 3 * SMPLModel::NUM_JOINTS);
    Matrix<double, 3, 1> offset; // a buffer for 3D vector
    Matrix<double, 3, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor> dtdP(3, 3 * SMPLModel::NUM_JOINTS); // a buffer for the derivative
    Matrix<double, 3, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor> dtdJ(3, 3 * SMPLModel::NUM_JOINTS); // a buffer for the derivative
    Matrix<double, 3, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor> dtdJ2(3, 3 * SMPLModel::NUM_JOINTS); // a buffer for the derivative

    std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> MR(SMPLModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(3, 3));
    std::vector<Eigen::Matrix<double, 3, 1>> Mt(SMPLModel::NUM_JOINTS, Eigen::Matrix<double, 3, 1>(3, 1));

    std::vector<Eigen::Matrix<double, 9, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor>> dMRdP(SMPLModel::NUM_JOINTS, Eigen::Matrix<double, 9, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor>(9, 3 * SMPLModel::NUM_JOINTS));
    std::vector<Eigen::Matrix<double, 3, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor>> dMtdP(SMPLModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor>(3, 3 * SMPLModel::NUM_JOINTS));
    std::vector<Eigen::Matrix<double, 3, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor>> dMtdJ(SMPLModel::NUM_JOINTS, Eigen::Matrix<double, 3, 3 * SMPLModel::NUM_JOINTS, Eigen::RowMajor>(3, 3 * SMPLModel::NUM_JOINTS));

    ceres::AngleAxisToRotationMatrix(pose, R.data());
    outJoint.row(0) = J0.row(0);
    MR.at(0) = R;
    Mt.at(0) = J0.row(0).transpose();
    outT.block(0, 0, 3, 3) = MR[0];
    outT.block(0, 3, 3, 1) = Mt[0];

    if (jacobians) {
        AngleAxisToRotationMatrix_Derivative(pose, dMRdP.at(0).data(), 0, numColumn);
        std::fill(dMtdP[0].data(), dMtdP[0].data() + 9 * SMPLModel::NUM_JOINTS, 0.0); // dMtdP.at(0).setZero();
        std::fill(dMtdJ[0].data(), dMtdJ[0].data() + 9 * SMPLModel::NUM_JOINTS, 0.0); // dMtdJ.at(0).setZero();
        dMtdJ.at(0).block(0, 0, 3, 3).setIdentity();
        std::copy(dMtdP[0].data(), dMtdP[0].data() + 9 * SMPLModel::NUM_JOINTS, dJdP.data()); // dJdP.block(0, 0, 3, SMPLModel::NUM_JOINTS * 3) = dMtdP[0];
        std::copy(dMtdJ[0].data(), dMtdJ[0].data() + 9 * SMPLModel::NUM_JOINTS, dJdJ.data()); // dJdJ.block(0, 0, 3, SMPLModel::NUM_JOINTS * 3) = dMtdJ[0];
    }
    for (int idj = 1; idj < mod_.NUM_JOINTS; idj++) {
        const int ipar = mod_.parent_[idj];
        const auto baseIndex = idj * 3;
        double angles[3] = { pose[baseIndex], pose[baseIndex + 1], pose[baseIndex + 2] };

        ceres::AngleAxisToRotationMatrix(angles, R.data());

        MR.at(idj) = MR.at(ipar) * R;
        offset = (J0.row(idj) - J0.row(ipar)).transpose();
        Mt.at(idj) = Mt.at(ipar) + MR.at(ipar) * offset;
        outJoint.row(idj) = Mt.at(idj).transpose();
        outT.block(0, 0, 3, 3) = MR[0];
        outT.block(0, 3, 3, 1) = Mt[0];
        if (jacobians) {

            AngleAxisToRotationMatrix_Derivative(angles, dRdP.data(), idj, numColumn);
            // Sparse derivative
            SparseProductDerivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), idj, mParentIndices.at(idj), dMRdP.at(idj).data(), numColumn);
            // // Slower but equivalent - Dense derivative
            // Product_Derivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), dMRdP.at(idj).data()); // Compute the product of matrix multiplication
            SparseProductDerivative(dMRdP.at(ipar).data(), offset.data(), mParentIndices.at(ipar), dMtdP.at(idj).data(), numColumn);
            // the following line is equivalent to dMtdP.at(idj) = dMtdP.at(idj) + dMtdP.at(ipar);
            SparseAdd(dMtdP.at(ipar).data(), mParentIndices.at(ipar), dMtdP.at(idj).data(), numColumn);

            std::fill(dtdJ.data(), dtdJ.data() + 9 * SMPLModel::NUM_JOINTS, 0.0); // dtdJ.setZero();
            // the following two lines are equiv to: dtdJ.block(0, 3 * idj, 3, 3).setIdentity(); dtdJ.block(0, 3 * ipar, 3, 3) -= Matrix<double, 3, 3>::Identity(); // derivative of offset wrt J
            dtdJ.data()[3 * idj] = 1;
            dtdJ.data()[3 * idj + 3 * SMPLModel::NUM_JOINTS + 1] = 1;
            dtdJ.data()[3 * idj + 6 * SMPLModel::NUM_JOINTS + 2] = 1;
            dtdJ.data()[3 * ipar] = -1;
            dtdJ.data()[3 * ipar + 3 * SMPLModel::NUM_JOINTS + 1] = -1;
            dtdJ.data()[3 * ipar + 6 * SMPLModel::NUM_JOINTS + 2] = -1;
            // the following line is equivalent to Product_Derivative(MR.at(ipar).data(), NULL, offset.data(), dtdJ.data(), dMtdJ.at(idj).data(), 1); // dA_data is NULL since rotation is not related to joint
            SparseProductDerivativeConstA(MR.at(ipar).data(), dtdJ.data(), mParentIndices.at(idj), dMtdJ.at(idj).data(), numColumn);
            // the following line is equivalent to dMtdJ.at(idj) = dMtdJ.at(idj) + dMtdJ.at(ipar);
            SparseAdd(dMtdJ.at(ipar).data(), mParentIndices.at(idj), dMtdJ.at(idj).data(), numColumn);

            std::copy(dMtdP[idj].data(), dMtdP[idj].data() + 9 * SMPLModel::NUM_JOINTS, dJdP.data() + 9 * idj * SMPLModel::NUM_JOINTS); // dJdP.block(3 * idj, 0, 3, SMPLModel::NUM_JOINTS * 3) = dMtdP[idj];
            std::copy(dMtdJ[idj].data(), dMtdJ[idj].data() + 9 * SMPLModel::NUM_JOINTS, dJdJ.data() + 9 * idj * SMPLModel::NUM_JOINTS); // dJdJ.block(3 * idj, 0, 3, SMPLModel::NUM_JOINTS * 3) = dMtdJ[idj];
        }
    }

    for (int idj = 0; idj < mod_.NUM_JOINTS; idj++) {
        offset = J0.row(idj).transpose();
        Mt.at(idj) -= MR.at(idj) * offset;

        outT.block(3 * idj, 0, 3, 3) = MR.at(idj);
        outT.block(3 * idj, 3, 3, 1) = Mt.at(idj);

        if (jacobians) {
            // The following line is equivalent to Product_Derivative(MR.at(idj).data(), dMRdP.at(idj).data(), offset.data(), NULL, dtdP.data(), 1);
            SparseProductDerivative(dMRdP.at(idj).data(), offset.data(), mParentIndices.at(idj), dtdP.data(), numColumn);
            // The following line is equivalent to dMtdP.at(idj) -= dtdP;
            SparseSubtract(dtdP.data(), mParentIndices.at(idj), dMtdP.at(idj).data(), numColumn);

            std::fill(dtdJ.data(), dtdJ.data() + 9 * SMPLModel::NUM_JOINTS, 0.0); // dtdJ.setZero();
            // The follwing line is equivalent to dtdJ.block(0, 3 * idj, 3, 3).setIdentity();
            dtdJ.data()[3 * idj] = 1;
            dtdJ.data()[3 * idj + 3 * SMPLModel::NUM_JOINTS + 1] = 1;
            dtdJ.data()[3 * idj + 6 * SMPLModel::NUM_JOINTS + 2] = 1;
            // The following line is equivalent to Product_Derivative(MR.at(idj).data(), NULL, offset.data(), dtdJ.data(), dtdJ2.data(), 1);
            SparseProductDerivativeConstA(MR.at(idj).data(), dtdJ.data(), mParentIndices.at(idj), dtdJ2.data(), numColumn);
            // The following line is equivalent to dMtdJ.at(idj) -= dtdJ2;
            SparseSubtract(dtdJ2.data(), mParentIndices.at(idj), dMtdJ.at(idj).data(), numColumn);

            // The following lines are copying jacobian from dMRdP and dMtdP to dTrdP, equivalent to
            // dTrdP.block(12 * idj + 0, 0, 3, SMPLModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(0, 0, 3, SMPLModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 4, 0, 3, SMPLModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(3, 0, 3, SMPLModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 8, 0, 3, SMPLModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(6, 0, 3, SMPLModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 3, 0, 1, SMPLModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(0, 0, 1, SMPLModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 7, 0, 1, SMPLModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(1, 0, 1, SMPLModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 11, 0, 1, SMPLModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(2, 0, 1, SMPLModel::NUM_JOINTS * 3);
            std::copy(dMRdP.at(idj).data(), dMRdP.at(idj).data() + 9 * SMPLModel::NUM_JOINTS, dTrdP.data() + 12 * idj * SMPLModel::NUM_JOINTS * 3);
            std::copy(dMtdP.at(idj).data(), dMtdP.at(idj).data() + 3 * SMPLModel::NUM_JOINTS, dTrdP.data() + (12 * idj + 3) * SMPLModel::NUM_JOINTS * 3);
            std::copy(dMRdP.at(idj).data() + 9 * SMPLModel::NUM_JOINTS, dMRdP.at(idj).data() + 18 * SMPLModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 4) * SMPLModel::NUM_JOINTS * 3);
            std::copy(dMtdP.at(idj).data() + 3 * SMPLModel::NUM_JOINTS, dMtdP.at(idj).data() + 6 * SMPLModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 7) * SMPLModel::NUM_JOINTS * 3);
            std::copy(dMRdP.at(idj).data() + 18 * SMPLModel::NUM_JOINTS, dMRdP.at(idj).data() + 27 * SMPLModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 8) * SMPLModel::NUM_JOINTS * 3);
            std::copy(dMtdP.at(idj).data() + 6 * SMPLModel::NUM_JOINTS, dMtdP.at(idj).data() + 9 * SMPLModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 11) * SMPLModel::NUM_JOINTS * 3);

            // The following lines are copying jacobian from and dMtdJ to dTrdJ, equivalent to
            // dTrdJ.block(12 * idj + 3, 0, 1, SMPLModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(0, 0, 1, SMPLModel::NUM_JOINTS * 3);
            // dTrdJ.block(12 * idj + 7, 0, 1, SMPLModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(1, 0, 1, SMPLModel::NUM_JOINTS * 3);
            // dTrdJ.block(12 * idj + 11, 0, 1, SMPLModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(2, 0, 1, SMPLModel::NUM_JOINTS * 3);
            std::copy(dMtdJ.at(idj).data(), dMtdJ.at(idj).data() + 3 * SMPLModel::NUM_JOINTS, dTrdJ.data() + (12 * idj + 3) * SMPLModel::NUM_JOINTS * 3);
            std::copy(dMtdJ.at(idj).data() + 3 * SMPLModel::NUM_JOINTS, dMtdJ.at(idj).data() + 6 * SMPLModel::NUM_JOINTS,
                dTrdJ.data() + (12 * idj + 7) * SMPLModel::NUM_JOINTS * 3);
            std::copy(dMtdJ.at(idj).data() + 6 * SMPLModel::NUM_JOINTS, dMtdJ.at(idj).data() + 9 * SMPLModel::NUM_JOINTS,
                dTrdJ.data() + (12 * idj + 11) * SMPLModel::NUM_JOINTS * 3);
        }
    }
}

using namespace Eigen;

void AngleAxisToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns)
{
    Eigen::Map<Eigen::Matrix<double, 9, Eigen::Dynamic, Eigen::RowMajor>> dR(dR_data, 9, numberColumns);
    std::fill(dR_data, dR_data + 9 * numberColumns, 0.0);
    const double theta2 = pose[0] * pose[0] + pose[1] * pose[1] + pose[2] * pose[2];
    if (theta2 > std::numeric_limits<double>::epsilon()) {
        const double theta = sqrt(theta2);
        const double s = sin(theta);
        const double c = cos(theta);
        const Eigen::Map<const Eigen::Matrix<double, 3, 1>> u(pose);
        Eigen::VectorXd e(3);
        e[0] = pose[0] / theta;
        e[1] = pose[1] / theta;
        e[2] = pose[2] / theta;

        // dR / dtheta
        Eigen::Matrix<double, 9, 1> dRdth(9, 1);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> dRdth_(dRdth.data());
        // skew symmetric
        dRdth_ << 0.0, -e[2], e[1],
            e[2], 0.0, -e[0],
            -e[1], e[0], 0.0;
        // dRdth_ = dRdth_ * c - Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();
        dRdth_ = -dRdth_ * c - Matrix<double, 3, 3>::Identity() * s + s * e * e.transpose();

        // dR / de
        Eigen::Matrix<double, 9, 3, RowMajor> dRde(9, 3);
        // d(ee^T) / de
        dRde << 2 * e[0], 0., 0.,
            e[1], e[0], 0.,
            e[2], 0., e[0],
            e[1], e[0], 0.,
            0., 2 * e[1], 0.,
            0., e[2], e[1],
            e[2], 0., e[0],
            0., e[2], e[1],
            0., 0., 2 * e[2];
        Eigen::Matrix<double, 9, 3, RowMajor> dexde(9, 3);
        dexde << 0, 0, 0,
            0, 0, -1,
            0, 1, 0,
            0, 0, 1,
            0, 0, 0,
            -1, 0, 0,
            0, -1, 0,
            1, 0, 0,
            0, 0, 0;
        // dRde = dRde * (1. - c) + c * dexde;
        dRde = dRde * (1. - c) - s * dexde;
        Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> dedu = Matrix<double, 3, 3>::Identity() / theta - u * u.transpose() / theta2 / theta;

        dR.block(0, 3 * idj, 9, 3) = dRdth * e.transpose() + dRde * dedu;
    } else {
        dR(1, 3 * idj + 2) = 1;
        dR(2, 3 * idj + 1) = -1;
        dR(3, 3 * idj + 2) = -1;
        dR(5, 3 * idj) = 1;
        dR(6, 3 * idj + 1) = 1;
        dR(7, 3 * idj) = -1;
    }
}

void Product_Derivative(const double* const A_data, const double* const dA_data, const double* const B_data,
    const double* const dB_data, double* dAB_data, const int B_col)
{
    assert(dA_data != NULL || dB_data != NULL);
    assert(B_col == 3 || B_col == 1); // matrix multiplication or matrix-vector multiplication
    if (dA_data != NULL && dB_data != NULL) {
        const Eigen::Map<const Eigen::Matrix<double, 9, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dA(dA_data);
        if (B_col == 1) {
            // B_col == 1
            // d(AB) = AdB + (dA)B
            const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> A(A_data);
            const Eigen::Map<const Eigen::Matrix<double, 3, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dB(dB_data);
            Eigen::Map<Eigen::Matrix<double, 3, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dAB(dAB_data);
            for (int r = 0; r < 3; r++) {
                const int baseIndex = 3 * r;
                dAB.row(r) = A(r, 0) * dB.row(0) + A(r, 1) * dB.row(1) + A(r, 2) * dB.row(2) + B_data[0] * dA.row(baseIndex) + B_data[1] * dA.row(baseIndex + 1) + B_data[2] * dA.row(baseIndex + 2);
            }
        } else {
            // B_col == 3
            // d(AB) = AdB + (dA)B
            const Eigen::Map<const Eigen::Matrix<double, 9, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dB(dB_data);
            Eigen::Map<Eigen::Matrix<double, 9, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dAB(dAB_data);
            for (int r = 0; r < 3; r++) {
                const int baseIndex = 3 * r;
                for (int c = 0; c < 3; c++) {
                    dAB.row(baseIndex + c) = A_data[baseIndex] * dB.row(c) + A_data[baseIndex + 1] * dB.row(3 + c) + A_data[baseIndex + 2] * dB.row(6 + c) + B_data[c] * dA.row(baseIndex) + B_data[3 + c] * dA.row(baseIndex + 1) + B_data[6 + c] * dA.row(baseIndex + 2);
                }
            }
        }
    } else if (dA_data != NULL && dB_data == NULL) // B is a constant matrix / vector, no derivative
    {
        const Eigen::Map<const Eigen::Matrix<double, 9, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dA(dA_data);
        if (B_col == 1) {
            // d(AB) = AdB + (dA)B
            Eigen::Map<Eigen::Matrix<double, 3, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dAB(dAB_data);
            // // Matrix form (slower)
            // for (int r = 0; r < 3; r++)
            //     dABAux.row(r) = B * dA.block<3, TotalModel::NUM_JOINTS * 3>(r, 0);
            // For loop form
            for (int r = 0; r < 3; r++) {
                const int baseIndex = 3 * r;
                dAB.row(r) = B_data[0] * dA.row(baseIndex) + B_data[1] * dA.row(baseIndex + 1) + B_data[2] * dA.row(baseIndex + 2);
            }
        } else {
            // B_col == 3
            Eigen::Map<Eigen::Matrix<double, 9, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dAB(dAB_data);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    dAB.row(3 * r + c) = B_data[c] * dA.row(3 * r) + B_data[3 + c] * dA.row(3 * r + 1) + B_data[6 + c] * dA.row(3 * r + 2); // d(AB) = AdB + (dA)B
        }
    } else // A is a constant matrix, no derivative
    {
        const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> A(A_data);
        // dA_data == NULL && dB_data != NULL
        if (B_col == 1) {
            const Eigen::Map<const Eigen::Matrix<double, 3, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dB(dB_data);
            Eigen::Map<Eigen::Matrix<double, 3, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dAB(dAB_data);
            dAB.setZero();
            for (int r = 0; r < 3; r++)
                dAB.row(r) = A(r, 0) * dB.row(0) + A(r, 1) * dB.row(1) + A(r, 2) * dB.row(2);
        } else {
            // B_col == 3
            const Eigen::Map<const Eigen::Matrix<double, 9, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dB(dB_data);
            Eigen::Map<Eigen::Matrix<double, 9, SMPLModel::NUM_JOINTS * 3, Eigen::RowMajor>> dAB(dAB_data);
            dAB.setZero();
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    dAB.row(3 * r + c) = A(r, 0) * dB.row(0 + c) + A(r, 1) * dB.row(3 + c) + A(r, 2) * dB.row(6 + c);
        }
    }
}

void SparseProductDerivative(const double* const A_data, const double* const dA_data, const double* const B_data,
    const double* const dB_data, const int colIndex, const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns)
{
    // d(AB) = AdB + (dA)B
    Eigen::Map<Eigen::Matrix<double, 9, Eigen::Dynamic, Eigen::RowMajor>> dAB(dAB_data, 9, numberColumns);

    std::fill(dAB_data, dAB_data + 9 * numberColumns, 0.0);
    // // Dense dAB (sparse dB) version
    // const Eigen::Map<const Eigen::Matrix<double, 9, numberColumns, Eigen::RowMajor> > dA(dA_data);
    // const Eigen::Map<const Eigen::Matrix<double, 9, numberColumns, Eigen::RowMajor> > dB(dB_data);
    // dAB.row(baseIndex + c) = B_data[c] * dA.row(baseIndex) + B_data[3 + c] * dA.row(baseIndex + 1) + B_data[6 + c] * dA.row(baseIndex + 2);
    // dAB.block<1,3>(baseIndex + c, 3*colIndex) += A_data[baseIndex] * dB.block<1,3>(c, 3*colIndex)
    //                                            + A_data[baseIndex+1] * dB.block<1,3>(3+c, 3*colIndex)
    //                                            + A_data[baseIndex+2] * dB.block<1,3>(6+c, 3*colIndex);
    // Sparse sped up equivalent
    const auto colOffset = 3 * colIndex;
    for (int r = 0; r < 3; r++) {
        const int baseIndex = 3 * r;
        for (int c = 0; c < 3; c++) {
            // AdB
            for (int subIndex = 0; subIndex < 3; subIndex++) {
                const auto finalOffset = colOffset + subIndex;
                dAB_data[numberColumns * (baseIndex + c) + finalOffset] += A_data[baseIndex] * dB_data[numberColumns * c + finalOffset]
                    + A_data[baseIndex + 1] * dB_data[numberColumns * (3 + c) + finalOffset]
                    + A_data[baseIndex + 2] * dB_data[numberColumns * (6 + c) + finalOffset];
            }
            // // AdB - Slower equivalent
            // dAB.block<1,3>(baseIndex + c, colOffset) += A_data[baseIndex] * dB.block<1,3>(c, colOffset)
            //                                            + A_data[baseIndex+1] * dB.block<1,3>(3+c, colOffset)
            //                                            + A_data[baseIndex+2] * dB.block<1,3>(6+c, colOffset);
            // (dA)B
            for (const auto& parentIndex : parentIndexes) {
                const auto parentOffset = 3 * parentIndex;
                for (int subIndex = 0; subIndex < 3; subIndex++) {
                    const auto finalOffset = parentOffset + subIndex;
                    dAB_data[numberColumns * (baseIndex + c) + finalOffset] += B_data[c] * dA_data[numberColumns * baseIndex + finalOffset]
                        + B_data[3 + c] * dA_data[numberColumns * (baseIndex + 1) + finalOffset]
                        + B_data[6 + c] * dA_data[numberColumns * (baseIndex + 2) + finalOffset];
                }
            }
            // // (dA)B - Slower equivalent
            // for (const auto& parentIndex : parentIndexes)
            // {
            //     const auto parentOffset = 3*parentIndex;
            //     dAB.block<1,3>(baseIndex + c, parentOffset) += B_data[c] * dA.block<1,3>(baseIndex, parentOffset)
            //                                                  + B_data[3 + c] * dA.block<1,3>(baseIndex+1, parentOffset)
            //                                                  + B_data[6 + c] * dA.block<1,3>(baseIndex+2, parentOffset);
            // }
        }
    }
}

void SparseProductDerivative(const double* const dA_data, const double* const B_data,
    const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns)
{
    // d(AB) = AdB + (dA)B
    // Sparse for loop form
    std::fill(dAB_data, dAB_data + 3 * numberColumns, 0.0);
    for (int r = 0; r < 3; r++) {
        const int baseIndex = 3 * r;
        for (const auto& parentIndex : parentIndexes) {
            const auto parentOffset = 3 * parentIndex;
            for (int subIndex = 0; subIndex < 3; subIndex++) {
                const auto finalOffset = parentOffset + subIndex;
                dAB_data[numberColumns * r + finalOffset] += B_data[0] * dA_data[numberColumns * baseIndex + finalOffset]
                    + B_data[1] * dA_data[numberColumns * (baseIndex + 1) + finalOffset]
                    + B_data[2] * dA_data[numberColumns * (baseIndex + 2) + finalOffset];
            }
        }
    }
    // // Dense Matrix form (slower)
    // Eigen::Map< Eigen::Matrix<double, 3, numberColumns, Eigen::RowMajor> > dAB(dAB_data);
    // const Eigen::Map<const Eigen::Matrix<double, 9, numberColumns, Eigen::RowMajor> > dA(dA_data);
    // for (int r = 0; r < 3; r++)
    //     dABAux.row(r) = B * dA.block<3, numberColumns>(r, 0);
    // // Dense for loop form
    // for (int r = 0; r < 3; r++)
    // {
    //     const int baseIndex = 3*r;
    //     dAB.row(r) = B_data[0] * dA.row(baseIndex) + B_data[1] * dA.row(baseIndex + 1) + B_data[2] * dA.row(baseIndex + 2);
    // }
}

void SparseProductDerivativeConstA(const double* const A_data, const double* const dB_data,
    const std::vector<int>& parentIndexes, double* dAB_data, const int numberColumns)
{
    // d(AB) = AdB (A is a constant.)
    // Sparse for loop form
    std::fill(dAB_data, dAB_data + 3 * numberColumns, 0.0);
    for (int r = 0; r < 3; r++) {
        for (const auto& parentIndex : parentIndexes) {
            const auto parentOffset = 3 * parentIndex;
            for (int subIndex = 0; subIndex < 3; subIndex++) {
                const auto finalOffset = parentOffset + subIndex;
                dAB_data[numberColumns * r + finalOffset] = A_data[3 * r + 0] * dB_data[finalOffset] + A_data[3 * r + 1] * dB_data[finalOffset + numberColumns] + A_data[3 * r + 2] * dB_data[finalOffset + numberColumns + numberColumns];
            }
        }
    }
}

void SparseAdd(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns)
{
    // d(AB) += d(AB)_parent
    Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> A(A_data, 3, numberColumns);
    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> B(B_data, 3, numberColumns);
    // Sparse for loop
    for (int r = 0; r < 3; r++) {
        for (const auto& parentIndex : parentIndexes) {
            const auto parentOffset = 3 * parentIndex;
            for (int subIndex = 0; subIndex < 3; subIndex++) {
                const auto finalOffset = parentOffset + subIndex;
                A_data[numberColumns * r + finalOffset] += B_data[numberColumns * r + finalOffset];
            }
        }
    }
    // // Dense equivalent
    // dMtdPIdj += dJdP.block<3, numberColumns>(3 * ipar, 0);
    // A += B;
}

void SparseSubtract(const double* const B_data, const std::vector<int>& parentIndexes, double* A_data, const int numberColumns)
{
    // d(AB) += d(AB)_parent
    Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> A(A_data, 3, numberColumns);
    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> B(B_data, 3, numberColumns);
    // Sparse for loop
    for (int r = 0; r < 3; r++) {
        for (const auto& parentIndex : parentIndexes) {
            const auto parentOffset = 3 * parentIndex;
            for (int subIndex = 0; subIndex < 3; subIndex++) {
                const auto finalOffset = parentOffset + subIndex;
                A_data[numberColumns * r + finalOffset] -= B_data[numberColumns * r + finalOffset];
            }
        }
    }
       // // Dense equivalent
    // dMtdPIdj -= dJdP.block<3, numberColumns>(3 * ipar, 0);
    // A -= B;
}