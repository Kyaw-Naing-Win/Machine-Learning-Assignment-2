import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class LinearRegression:
    def __init__(self, init_theta, alpha=0.01, method='batch', momentum=0.0, regularization=None,
                 num_epochs=500, batch_size=100, cv=None):
        
        self.init_theta_mode = init_theta  
        self.alpha = alpha
        self.method = method
        self.momentum = momentum 
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cv = cv if cv is not None else KFold(n_splits=5) 

        self.theta = None
        self.prev_step = 0
        

    def fit(self, X_train, y_train):
        self.old_val_loss = np.inf
        self.kfold_loss_scores = []
        self.kfold_r2_scores = []
        best_theta = None  

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            print(f"\n--- Fold {fold + 1} ---")

            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            
            if self.init_theta_mode == 'zero':
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.init_theta_mode == 'xavier':
                m = X_cross_train.shape[1] 
                bound = 1.0 / np.sqrt(m)
                self.theta = np.random.uniform(-bound, bound, size=(m,))
            else:
                raise ValueError("Unsupported init_theta. Use 'zero' or 'xavier'.")

            self.prev_step = 0  

            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):

                for epoch in range(self.num_epochs):
                    # Shuffle data
                    perm = np.random.permutation(X_cross_train.shape[0])
                    X_method_train = X_cross_train[perm]
                    y_method_train = y_cross_train[perm]

                    # Unified batch handling
                    if self.method == 'sto':
                        for i in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[i].reshape(1, -1)
                            y_method_train = y_cross_train[i]
                            train_loss = self._train(X_method_train, y_method_train)

                    elif self.method == 'mini_batch':
                        for i in range(0, X_cross_train.shape[0], self.batch_size):
                            X_method_train= X_cross_train[i:i + self.batch_size]
                            y_method_train = y_cross_train[i:i + self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)

                    elif self.method == 'batch':
                        train_loss = self._train(X_cross_train, y_cross_train)

                    mlflow.log_metric(key='train_loss', value=train_loss, step = epoch)

                    # Evaluate on validation set
                    y_val_pred = self._predict(X_cross_val)
                    val_mse = self._mse(y_val_pred, y_cross_val)
                    mlflow.log_metric(key='val_mse_loss',value= val_mse,step= epoch)

                    val_r2 = self._r2_score(y_val_pred, y_cross_val)
                    mlflow.log_metric(key='val_r2',value= val_r2,step= epoch)

                    # Early stopping check
                    if np.allclose(val_mse, self.old_val_loss):
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                    self.old_val_loss = val_mse

                    
                
                self.kfold_loss_scores.append(val_mse)
                self.kfold_r2_scores.append(val_r2)

                print(f"Fold {fold} MSE: {val_mse}")
                print(f"Fold {fold} R2: {val_r2}")
                
        print("\nTraining complete.")

    def _train(self, X, y):
        y_pred = X @ self.theta
        error = y_pred - y
        gradient = (1 / X.shape[0]) * X.T @ error

        # Apply regularization
        if self.regularization is not None:
            gradient += self.regularization.derivation(self.theta)

        # Update using momentum
        step = self.alpha * gradient
        self.theta = self.theta - step + self.momentum * self.prev_step
        self.prev_step = step
        return self._mse(y_pred,y)

    def _predict(self, X):
        return X @ self.theta  

    def _mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def _avg_mse(self):
        return np.sum(np.array(self.kfold_loss_scores))/len(self.kfold_loss_scores)

    def _avg_r2_score(self):
        return np.sum(np.array(self.kfold_r2_scores))/len(self.kfold_r2_scores)

    def _r2_score(self, y_pred, y_true):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    def _coef(self):
        return self.theta[1:]

    def _intercept(self):
        return self.theta[0]

    def feature_importance(self, feature_names):
        # Fixed: method name typo, added feature_names as argument
        if self.theta is None:
            raise ValueError("Model is not trained yet.")

        importances = self._coef()
        sorted_idx = np.argsort(importances)

        # Fixed: spelling of 'columns', use feature_names input
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Linear Regression Feature Importances")
        plt.show()

#Lasso
class LassoPenalty:
    def __init__(self, l): #l = lamda
        self.l = l
        
    def __call__(self, init_theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(init_theta))
    
    def derivation(self, init_theta):
        return self.l * np.sign(init_theta)
    

#inherits LinearRegression and has separate classes for each of this regularization algorithm

class Lasso(LinearRegression):
    def __init__(self, init_theta, alpha,method, momentum,  l): # lr = learning rate, l = lamda
        self.regularization = LassoPenalty(l)
        super().__init__(init_theta, alpha,method, momentum,self.regularization)


#Ridge
class RidgePenalty:
    def __init__(self, l):
        self.l = l
        
    def __call__(self, init_theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(init_theta))
    
    def derivation(self, init_theta):
        return self.l * 2 * init_theta
    
class Ridge(LinearRegression):
    def __init__(self, init_theta, alpha,method, momentum, l):
        self.regularization = RidgePenalty(l)
        super().__init__(init_theta, alpha,method, momentum,self.regularization)


class NoRegularization:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, init_theta): #__call__ allows us to call class as method
        return 0
        
    def derivation(self, init_theta): # return 0, since we won't have any regularization.
        return 0
    
class Normal(LinearRegression):
    # self, init_theta, alpha,method, momentum, l
    def __init__(self, init_theta, alpha, method, momentum,l):
        self.regularization = NoRegularization(l)
        super().__init__(init_theta, alpha,method, momentum,self.regularization)