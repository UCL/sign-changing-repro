from ngsolve import *
import numpy as np
import sys
import scipy.sparse as sp
import pypardiso

# The function "delete_from_csr" is from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices?rq=3
def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat

# We define some more auxiliary functions to make use of the pypardiso solver 
def GetFreedofsList(freedofs):
    free_dofs = []
    non_free_dofs = []
    for nr,b in enumerate(freedofs):
        if not b:
            non_free_dofs.append(nr)
        else:
            free_dofs.append(nr) 
    return free_dofs,non_free_dofs

def GetSpMat(aX,non_free_dofs):
    rows,cols,vals = aX.mat.COO()
    Asp = sp.csr_matrix((vals,(rows,cols)))
    Asp_clean = delete_from_csr(Asp, row_indices=non_free_dofs, col_indices=non_free_dofs)
    return Asp_clean

class PySolver:
    def __init__(self,Asp,psolver):
        self.Asp = Asp
        self.solver = psolver
    def solve(self,b_inp,x_out):
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp)
        self.solver.set_phase(33)
        x_out[:] = self.solver._call_pardiso(self.Asp , b )[:]
    def __del__(self):
        del self.solver
        del self.Asp 

def GetPyPardisoSolver(mat,non_free_dofs):
    solver_instance = pypardiso.PyPardisoSolver()
    matrix_A  = GetSpMat(mat,non_free_dofs)
    solver_instance.factorize(matrix_A)
    return PySolver( matrix_A, solver_instance)
