

def check_model_drift(current_auc, threshold=0.7):
    if current_auc < threshold:
        print("⚠️ ALARM: Model performance degradation detected")
        return True
    return False
