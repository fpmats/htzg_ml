from htzg_utilities import *
def main():
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

    # loads full data 
    X_train_full = np.load('ml_data/Full_Data/split/X_train_full.npy')
    X_test_full = np.load('ml_data/Full_Data/split/X_test_full.npy')
    y_train_full = np.load('ml_data/Full_Data/split/y_train_full.npy')
    y_test_full = np.load('ml_data/Full_Data/split/y_test_full.npy')
    # loads reduced data
    X_train_redu = np.load('ml_data/Reduced_Data/split/X_train_redu.npy')
    X_test_redu = np.load('ml_data/Reduced_Data/split/X_test_redu.npy')
    y_train_redu = np.load('ml_data/Reduced_Data/split/y_train_redu.npy')
    y_test_redu = np.load('ml_data/Reduced_Data/split/y_test_redu.npy')

    final_model_redu = pickle.load(open('models/final_model_redu.sav', 'rb'))
    final_model_full = pickle.load(open('models/final_model_full.sav', 'rb'))

    evaluate_model(final_model_full,X_train_full,y_train_full,X_test_full,y_test_full,'Full Model:')
    evaluate_model(final_model_redu,X_train_redu,y_train_redu,X_test_redu,y_test_redu,'Reduced Model:')

    # Uses models for final predictions
    y_train_guess_full = final_model_full.predict(X_train_full)
    y_train_guess_redu = final_model_redu.predict(X_train_redu)
    y_test_guess_full = final_model_full.predict(X_test_full)
    y_test_guess_redu = final_model_redu.predict(X_test_redu)

    plot_pred_v_truth(y_test_full,y_train_full,y_test_guess_full,y_train_guess_full,
                  name = 'Full Model',split=.2,path='')
    plot_pred_v_truth(y_test_redu,y_train_redu,y_test_guess_redu,y_train_guess_redu,
                  name = 'Reduced Model',split=.2,path='')
    
    # loads full data 
    X_661_Final = np.load('ml_data/Full_Data/whole/X_661_Final.npy')
    X_661_Final_Reduced = (np.load('ml_data/Reduced_Data/whole/X_661_Final_Reduced.npy'))
    Y_661_Final = np.load('ml_data/Full_Data/whole/Y_661_Final.npy')

    generate_shap_beehive(final_model_full,X_661_Final,Y_661_Final,is_full=True)
    generate_shap_beehive(final_model_redu,X_661_Final_Reduced,Y_661_Final,is_full=False)
    generate_shap_avg(final_model_full,X_661_Final,Y_661_Final,is_full=True)
    generate_shap_avg(final_model_redu,X_661_Final_Reduced,Y_661_Final,is_full=False)
if __name__ == '__main__':
    sys.exit(main()) 