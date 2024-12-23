<style>
body {
    font-family: CodeNewRoman Nerd Font; /*Sorry for this, but I really like this font.*/
}
</style>

# GPRcpp
This is a simple implemention of Gaussian Process Regression in C++. Most of the code is translate from sklearn library and GPy library.
See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html and https://gpy.readthedocs.io/en/deploy/ for reference.

## Question
LDLT do no work, because of the P matrix. In LDLT, we get P^T * L * D * L^T * P = A, here L is lower triangular matrix.
Before this I naively think I can get LLT docomp of A from LDLT decomp, L_llt = P^T * L * D^{1/2}, but the P matrix will make this
matrix to non-triangular. And however you put P around L * D^{1/2} will not give you a triangular matrix, and L * D^{1/2} * (L * D^{1/2})^T 
does not equal to A.

Why I have to use LDLT? Because in aarch64 system the llt is not working on large matrix(and LDLT decomp works fine), but I need the lower triangular matrix l to do some calculation, so I stuck here. Maybe do a llt on P * A * P^T will give me better answer.