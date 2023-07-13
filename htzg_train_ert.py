from htzg_utilities import *
def main():
    # Warning - this will take a long time to run full hyperparameter optimization. 
    # If you want, you can set >use_bayes in the >train_final_model method to True
    # which will use a Baysian hyperparameter search instead of a grid search, which
    # is more efficent at the risk of converging to a less optimal parameter set. 
    X_661_Final = np.load('ml_data/Full_Data/whole/X_661_Final.npy')
    X_661_Final_Reduced = (np.load('ml_data/Reduced_Data/whole/X_661_Final_Reduced.npy'))
    Y_661_Final = np.load('ml_data/Full_Data/whole/Y_661_Final.npy')

    X_661_Final,_ = remove_annoying_variables(X_661_Final,Y_661_Final)
    X_661_Final_Reduced,Y_661_Final = remove_annoying_variables(X_661_Final_Reduced,Y_661_Final)


    names = ['epsilons',
    'ens',
    'becs',
    'mean_atomic_masses',
    'mean_ph_freqs',
    'first_optical_ph_freqs',
    'eq_gaps',
    'atoms_per_areas',
    'areal_densities']

    reduced_names = ['ens','mean_atomic_masses',  'eq_gaps','atoms_per_areas',
    'areal_densities']


    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split( X_661_Final, Y_661_Final, test_size=0.2, random_state=0)
    X_train_redu, X_test_redu, y_train_redu, y_test_redu = train_test_split(X_661_Final_Reduced, Y_661_Final, test_size=0.2, random_state=0)
    # saves full data 
    np.save('ml_data/Full_Data/split/X_train_full',X_train_full)
    np.save('ml_data/Full_Data/split/X_test_full',X_test_full)
    np.save('ml_data/Full_Data/split/y_train_full',y_train_full)
    np.save('ml_data/Full_Data/split/y_test_full',y_test_full)
    # saves reduced data
    np.save('ml_data/Reduced_Data/split/X_train_redu',X_train_redu)
    np.save('ml_data/Reduced_Data/split/X_test_redu',X_test_redu)
    np.save('ml_data/Reduced_Data/split/y_train_redu',y_train_redu)
    np.save('ml_data/Reduced_Data/split/y_test_redu',y_test_redu)

    final_model_full = train_final_model(X_train_full,y_train_full,is_full=True,use_bayes=False)
    final_model_redu = train_final_model(X_train_redu,y_train_redu,is_full=False,use_bayes=False)

    pickle.dump(final_model_full, open('models/final_model_full.sav', 'wb'))
    pickle.dump(final_model_redu, open('models/final_model_redu.sav', 'wb'))
if __name__ == '__main__':
    sys.exit(main()) 