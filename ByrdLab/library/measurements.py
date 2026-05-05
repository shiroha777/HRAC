import torch
from sklearn.metrics import confusion_matrix
from ByrdLab import TARGET_TYPE, DEVICE

@torch.no_grad()
def binary_classification_accuracy(predictions, targets):
    '''
    Calculate binary classification accuracy.
    predictions: sigmoid output (values between 0 and 1), shape: (batch_size,)
    targets: binary labels (0 or 1), shape: (batch_size,)
    Returns: number of correct predictions as a tensor (to be accumulated and divided by total samples)
    '''
    # Convert predictions to binary (0 or 1) using threshold 0.5
    predictions_binary = (predictions > 0.5).type_as(targets)
    # Return number of correct predictions as tensor
    return (predictions_binary == targets).sum()

@torch.no_grad()
def multi_classification_accuracy(predictions, targets):
    '''
    Calculate multi-class classification accuracy.
    predictions: model output logits (before softmax), shape: (batch_size, num_classes)
    targets: class labels, shape: (batch_size,)
    Returns: number of correct predictions as a tensor (to be accumulated and divided by total samples)
    '''
    # Get predicted class indices (argmax)
    _, prediction_cls = torch.max(predictions.detach(), dim=1)
    # Return number of correct predictions as tensor
    return (prediction_cls == targets).sum()

@torch.no_grad()
def consensus_error(local_models, honest_nodes):
    return torch.var(local_models[honest_nodes], 
                     dim=0, unbiased=False).norm().item()

@torch.no_grad()
def avg_loss(model, get_test_iter, loss_fn, weight_decay):
    '''
    The function calculating the loss.
    '''
    loss = 0
    total_sample = 0
    
    # evaluation
    test_iter = get_test_iter()
    for features, targets in test_iter:
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        total_sample += len(targets)
    
    loss_avg = loss / total_sample
    for param in model.parameters():
        loss_avg += weight_decay * param.norm()**2 / 2

    return loss_avg
    
@torch.no_grad()
def avg_loss_accuracy(model, get_test_iter, loss_fn, weight_decay):
    '''
    The function calculating the loss and accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    test_iter = get_test_iter()
    for features, targets in test_iter:
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        _, prediction_cls = torch.max(predictions.detach(), dim=1)
        accuracy += (prediction_cls == targets).sum().item()
        total_sample += len(targets)
    
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample
    for param in model.parameters():
        loss_avg += weight_decay * param.norm()**2 / 2

    return loss_avg, accuracy_avg

@torch.no_grad()
def avg_loss_accuracy_dist(dist_models, get_test_iter,
                           loss_fn, test_fn, weight_decay,
                           node_list=None):
    '''
    The function calculating the loss and accuracy in distributed setting.
    Return the avarage of the local models' accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    # dist_models.activate_avg_model()
    if node_list is None:
        node_list = range(dist_models.node_size)
    for node in node_list:
        print(f'Node: {node}')
        dist_models.activate_model(node)
        model = dist_models.model
        model.eval()
        test_iter = get_test_iter()
        for features, targets in test_iter:
            # model.to(DEVICE)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            predictions = model(features)
            loss += loss_fn(predictions, targets).item()
            if test_fn is not None:
                accuracy += test_fn(predictions, targets).item()
            total_sample += len(targets)
        # penalization = 0
        # for param in model.parameters():
        #     penalization += weight_decay * param.norm()**2 / 2
        # loss += penalization
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample

    return loss_avg, accuracy_avg

@torch.no_grad()
def avg_loss_accuracy_dist_1(dist_models, get_test_iter,
                           loss_fn, test_fn, weight_decay,
                           node_list=None):
    '''
    The function calculating the loss and accuracy in distributed setting.
    Return the avarage of the local models' accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    # dist_models.activate_avg_model()
    if node_list is None:
        node_list = range(dist_models.node_size)
    #for node in node_list:

    #model.to(device)
    with torch.no_grad():
        node = node_list[0]
        dist_models.activate_model(node)
        model = dist_models.model
        model.eval() # very important, without , the loss and accuracy in test will worse
        test_iter = get_test_iter()
        for features, targets in test_iter:
            # model.to(device) #修改处
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            predictions = model(features)
            loss += loss_fn(predictions, targets)#.cpu().item()
            accuracy += test_fn(predictions, targets)#.cpu().item()
            total_sample += len(targets)
        
        # penalization = 0
        # for param in model.parameters():
        #     penalization += weight_decay * param.norm()**2 / 2
        # loss += penalization
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample

    return loss_avg, accuracy_avg



@torch.no_grad()
def one_node_loss_accuracy_dist(model, get_test_iter,
                           loss_fn, test_fn, weight_decay,
                           node_list=None):
    '''
    The function calculating the loss and accuracy in distributed setting.
    Return the avarage of the local models' accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    with torch.no_grad():
        model.eval()
        test_iter = get_test_iter()
        for features, targets in test_iter:
            # model.to(DEVICE)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            predictions = model(features)
            loss += loss_fn(predictions, targets)# .item()
            if test_fn is not None:
                accuracy += test_fn(predictions, targets)# .item()
            total_sample += len(targets)
            # penalization = 0
            # for param in model.parameters():
            #     penalization += weight_decay * param.norm()**2 / 2
            # loss += penalization
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample

    return loss_avg, accuracy_avg
