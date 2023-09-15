import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import subprocess  # Import the subprocess module
import mlflow 
import numpy as np
import math
from sklearn.model_selection import KFold

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3) # number of splites the model runs everytime. 
            
    def __init__(self, regularization, lr, method, theta_init, momentum, num_epochs=500, batch_size=50, cv=kfold): # linear regression class with parameters 
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.theta_init = theta_init
        self.momentum = momentum
        self.cv         = cv
        self.regularization = regularization

    def mse(self, ytrue, ypred): #defining function for Mean Sqaured Method funtion
        return ((ytrue - ypred) ** 2).sum() / ypred.shape[0]
    
    def r2(self, ytrue, ypred): #Defining function for R-Sqaured Funtion 
        return 1 - ((((ytrue - ypred) ** 2).sum()) / (((ytrue - ytrue.mean()) ** 2).sum()))
    
    def fit(self, X_train, y_train): #Fit method implements training the linear regression model 
            
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty
        # mlflow.log_params(params=params)  # THEL LINE U CHANGED 

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            if self.theta_init == 'zeros':
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.theta_init == 'xavier':
                m = X_train.shape[0]
                # calculate the range for the weights
                lower , upper = -(1.0 / math.sqrt(m)), (1.0 / math.sqrt(m))
                # summarize the range
                print(lower , upper)
                # you need to basically randomly pick weights within this range
                # generate random numbers
                numbers = np.random.rand(X_cross_train.shape[1])
                self.theta = lower + numbers * (upper - lower)
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            else:
                self.theta = np.zeros(X_cross_train.shape[1])
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__} # THE LINK BELOW U MOVED UP 
                mlflow.log_params(params=params) 
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    #Now we choose different gradient methods for training the model 
                    # 1st I have chosen Stochastic gradient descent 'sto'
                    # 2nd Mini-Batch Gradient 'mini'
                    # Each method updates the model parameters using different subsets for training each data set. 
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #record dataset
                    # mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
                    # mlflow.log_input(mlflow_train_data, context="training")
                    
                    # mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
                    # mlflow.log_input(mlflow_val_data, context="validation")
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
            
                    
    def _train(self, X, y): # This method is responsible for performing one step of gradient decent to update the model parameter
        yhat = self.predict(X) # two arguement x and y taking out true target value. 
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta) # computes gradient of loss funtion
        prev_step = 0 #implements momentum in gradient descent, initialized to 0 
        
        if self.momentum == "without":
            step = self.lr * grad
        else:
            step = self.lr * grad + self.momentum * prev_step

        self.theta -= step # 
        prev_step = step
        return self.mse(y, yhat)
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    # Feature Analysis 
    def plot_feature_importance(self, feature_names):
        import matplotlib.pyplot as plt
        feature_importance = np.abs(self._coef())
        sorted_indices = np.argsort(feature_importance)
        sorted_features = [feature_names[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, feature_importance[sorted_indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.show()
        
# Creating class for NormalPenalty defines a placeholder for L2 regularization where the regularization is 0
class NormalPenalty:
    
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return 0
        
    def derivation(self, theta):
        return 0

#Defining class which is known as L1 regularization helps in sparsity in the model by penalizing the absolute values of the parameters.
class LassoPenalty:
    
    def __init__(self, l): # the init is a conductor when an instance of a class is created 
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
#   Defining class. calculates the loss funtion during training for smaller values to avoid over fitting  
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
# This class intends for implementing Elastic net regularization
# combination of L1 (Lasso) & L2 (Ridge)
# Helps in sparsity in the model, handles multicollinearity    
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta): # class to calculate the gradient of a funtion with a set of parameters 
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
#This class inherits all the attributes and methods of the linear regression class      
    
class Normal(LinearRegression):
    
    def __init__(self, method, lr, theta_init, momentum, l):
        # self.regularization = NormalPenalty(l)
        super().__init__('normal', lr, method, theta_init, momentum) 
        # super().__init__(self.regularization, lr, method, theta_init, momentum)
        self.regularization = NormalPenalty(l)

#This class inherites all the attributes via modifing functionality specific to the lasso class
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, theta_init, momentum, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, theta_init, momentum)

#This class helps in accessing and using attributes methods of a parent class including its flexibility to override or extend         
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, theta_init, momentum, l): # momentum is a hyperparameter used for algorithms in gradient decent-based methods.
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, theta_init, momentum)


# This class is a model that combines L1 lasso & Ridge to achieve a balance features selection and handling multicollinearity.       
class ElasticNet(LinearRegression):
    
    def __init__(self, method, lr,theta_init, momentum, l, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, theta_init, momentum)


# Load the models from the .model files
model_a1 = joblib.load('car_price_predictionA1.model')
model_a2 = joblib.load('car_price_predictionA2.model')

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define styles for different elements (same as in your code)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Navigation links to different pages
    html.Div([
        dcc.Link('A1', href='/page-1'),
        html.Br(),
        dcc.Link('A2', href='/page-2'),
    ]),
    
    # Adding a hyperlink
    html.Div(id='page-content')        
])

page_1_layout = html.Div([
    html.H1("Car Price Prediction A1"),
    
    html.Label("Mileage (km)"),
    dcc.Input(id='input-mileage-1', type='number'),
    html.Label("Engine Size (cc)"),
    dcc.Input(id='input-engine-size-1', type='number'),
    html.Label("Max Power (hp)"),
    dcc.Input(id='input-max-power-1', type='number'),
    
    html.Button("Submit", id="Submit-button-1", n_clicks=0),
    html.Div(id='predicted-price-1'),
])


page_2_layout = html.Div([
    html.H1("Car Price Prediction A2"),
    
    html.Label("Mileage (km)"),
    dcc.Input(id='input-mileage-2', type='number'),
    html.Label("Engine Size (cc)"),
    dcc.Input(id='input-engine-size-2', type='number'),
    html.Label("Max Power (hp)"),
    dcc.Input(id='input-max-power-2', type='number'),
    
    html.Button("Submit", id="Submit-button-2", n_clicks=0),
    html.Div(id='predicted-price-2'),

    # Add a button to open Windows Prompt
    html.Button("PRESS FOR INFORMATION", id="open-prompt-button", n_clicks=0),
    html.Div(id='prompt-output'),
    
   
])

@app.callback(
    Output('predicted-price-1', 'children'),
    [Input('Submit-button-1', 'n_clicks')],
    [State('input-mileage-1', 'value'),
     State('input-engine-size-1', 'value'),
     State('input-max-power-1', 'value')]
)
def update_predicted_price_1(n_clicks, mileage, engine_size, max_power):
    if n_clicks is not None and n_clicks > 0:
        input_data = [mileage, engine_size, max_power]
        predicted_price = model_a1.predict([input_data])[0]
        return f'Predicted Price A1: ${predicted_price:.2f}'
    return ""

@app.callback(
    Output('predicted-price-2', 'children'),
    [Input('Submit-button-2', 'n_clicks')],
    [State('input-mileage-2', 'value'),
     State('input-engine-size-2', 'value'),
     State('input-max-power-2', 'value')]
)
def update_predicted_price_2(n_clicks, mileage, engine_size, max_power):
    if n_clicks is not None and n_clicks > 0:
        input_data = [mileage, engine_size, max_power]
        predicted_price = model_a2.predict([input_data])[0]
        return f'Predicted Price A2: ${predicted_price:.2f}'
    return ""

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page-2':
        return page_2_layout
    else:
        return page_1_layout

# Callback to open Windows Prompt

@app.callback(
    Output('prompt-output', 'children'),
    [Input('open-prompt-button', 'n_clicks')]
)
def open_windows_prompt(n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1:
        message = (
            "Use this new model to make car price predictions based on the provided inputs."
            " Fill in the values for max power, mileage, and engine size, and click 'Submit' "
            "to see the predicted car price."
            " You will realize that the new model is more accurate and better. Enjoy :)"
        )
        return html.Div(
            [html.P(message, style={"fontSize": 16, "textAlign": "center", "color": "#333"})]
        )
    return ""



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
    