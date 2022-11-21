from sklearn.metrics import accuracy_score
y_pred = ['a', 'b', 'b']
y_real = ['a', 'a', 'b']
if __name__ == "__main__":
    print(accuracy_score(y_real,y_pred))