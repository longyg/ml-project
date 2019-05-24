import train_mlp_model
import load_data

import matplotlib.pyplot as plt
from matplotlib import cm


def tune_mlp_model():
    _, data = load_data.load_cook_train_data(isLemmatize=True)
    
    num_layers = [1, 2, 3]
    num_units = [8, 16, 32, 64, 128]
    dropout_rates = [0.1, 0.2, 0.3, 0.4]

    params = {
        # 'layers': [],
        # 'units': [],
        'dropout_rate': [],
        'accuracy': []
    }

    # for layers in num_layers:
    #     for units in num_units:
    for dropout_rate in dropout_rates:
        # params['layers'] = layers
        # params['units'] = units
        params['dropout_rate'] = dropout_rate

        accuracy, _ = train_mlp_model.train_mlp_model(data, 
                                                    #   units=units, 
                                                    #   layers=layers,
                                                      dropout_rate=dropout_rate)
        # print('Accuracy: {accuracy}, Parameters: (layers={layers}, '
        #         'units={units})'.format(accuracy=accuracy, units=units, layers=layers))
        print('Accuracy: {accuracy}, Parameters: dropout_rate={dropout_rate}'.format(accuracy=accuracy, dropout_rate=dropout_rate))
        params['accuracy'] = accuracy
    _plot_parameters(params)

def _plot_parameters(params):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(params['layers'],
                    params['units'],
                    params['accuracy'],
                    cmap=cm.coolwarm,
                    antialiased=False)
    plt.show()
    
if __name__ == '__main__':
    tune_mlp_model()