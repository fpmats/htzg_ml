from htzg_utilities import *
def main():
    # all linear regression stuff
    X_661_Final = np.load('ml_data/Full_Data/whole/X_661_Final.npy')
    X_661_Final_Reduced = (np.load('ml_data/Reduced_Data/whole/X_661_Final_Reduced.npy'))
    Y_661_Final = np.load('ml_data/Full_Data/whole/Y_661_Final.npy')
    X_661_Final,_ = remove_annoying_variables(X_661_Final,Y_661_Final)
    X_661_Final_Reduced,Y_661_Final = remove_annoying_variables(X_661_Final_Reduced,Y_661_Final)

    # full model
    (full_r2,full_weights,full_y_pred) = train_linear_regression(X_661_Final,Y_661_Final)
    plot_regression(Y_661_Final,full_y_pred,name = 'Full',path='plots')
    print('*** Full Model Linear Regression Weights *** ')
    print(full_r2,full_weights)
    print(['epsilons',
                    'ens',
                    'becs',
                    'mean_atomic_masses',
                    'mean_ph_freqs',
                    'first_optical_ph_freqs',
                    'eq_gaps',
                    'atoms_per_areas',
                    'areal_densities'])
    # reduced model
    (redu_r2,redu_weights,redu_y_pred) = train_linear_regression(X_661_Final_Reduced,Y_661_Final)
    plot_regression(Y_661_Final,redu_y_pred,name = 'Reduced',path='plots')
    print('*** Full Model Linear Regression Weights *** ')
    print(redu_r2,redu_weights)
    print(['ens','mean_atomic_masses',  'eq_gaps','atoms_per_areas','areal_densities'])
if __name__ == '__main__':
    sys.exit(main()) 