<style>
body {
    font-family: CodeNewRoman Nerd Font;
}
</style>

## TODO LIST (Clean at 2024.11.12)
- [ ] gpr_test.cpp test_with_big_data function bug with ldlt
- [x] Propagete the uncertainty.
- [ ] Propagete the covariance using equation 35 in reference [1].
- [ ] maybe realtime update model
- [x] Use the compute and solve in llt
- [x] Implement FITC method
- [x] Make it good for graduation project

## Change Log
- 2024.12.23: 
    - dk_dx implemented, but new input should only be 1 sample.
    - Propagete the uncertainty, without covariance and 2-d variance, input should be 1 sample.
- 2024.11.13: FITC inference method implemented
- 2024.11.12:
    - Add support for the importing of tranning data in a fixed way
    - var DTC method implemented
- 2024.11.8: add GPy compare script file
- 2024.11.7: add normalization support

## References
1. A. Girard, C. E. Rasmussen, and R. Murray-Smith, “Gaussian Process priors with Uncertain Inputs: Multiple-Step-Ahead Prediction”.