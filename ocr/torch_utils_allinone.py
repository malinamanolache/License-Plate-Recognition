import torch
import torch.nn as nn
from tqdm import tqdm
import os



def compute_class_weights(labels, num_classes, device):
    # Flatten labels to count class occurrences
    labels_flat = labels.argmax(dim=2).view(-1)
    class_counts = torch.bincount(labels_flat, minlength=num_classes)
    
    # Inverse frequency for weights
    weights = 1.0 / (class_counts.float() + 1e-9)
    #weights = class_counts.float()
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return weights.to(device)

def iterate(model, 
            dataloader, 
            optimizer, 
            loss_fn, 
            loss_fn_plate_type,
            is_training=True, 
            device='cuda'):
    """
    Single-iteration function, either for training or for testing.

    :param model: nn.Module, neural network
    :param dataloader: torch.utils.data.DataLoader object to iterate over
    :param optimizer: torch.optim object, used to update the model if is_training=True
    :param loss_fn: cost function
    :param is_training: bool, whether this iteration should compute gradients and update the model
    """
    # Set the model to training mode if it's a training step, otherwise to evaluation mode
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0
    total_acc = 0
    total_no_correct_preds_type = 0
    total_samples = 0

    num_classes = 36  # Number of character classes (26 letters + 10 digits)
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)

    for inputs, labels, labels_string, plate_type in tqdm(dataloader):
        if isinstance(inputs, list):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device)
        else:
            inputs = inputs.to(device)
        
        if isinstance(labels, list):
            for i in range(len(labels)):
                labels[i] = labels[i].to(device)
        else:
            labels = labels.to(device)

        if isinstance(plate_type, list):
            for i in range(len(plate_type)):
                plate_type[i] = plate_type[i].to(device)
        else:
            plate_type = plate_type.to(device)
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # VERY IMPORTANT: decomment here to get the weighted loss fucntion
        # comment to go back to the unweighed version
        #class_weights = compute_class_weights(labels, num_classes, device)
        #loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # Forward pass
        output_classes, output_plate_type = model(inputs)

        # Calculate loss if it's a training step
        classes_loss = 0
        for i in range(7):  # Loop over the sequence length
            classes_loss += loss_fn(output_classes[:, i, :], labels[:, i, :])

        classes_loss /= 7  # Average the loss over the sequence length

        plate_type_loss = loss_fn_plate_type(output_plate_type, plate_type.unsqueeze(1).float())

        # Compute accuracy
        _, preds = torch.max(output_classes, dim=2)  # preds shape will be [batch_size, number_of_classified_objects]
        labels_indices = labels.argmax(dim=2)

        # Calculate the number of correct predictions for each class
        for class_idx in range(num_classes):
            class_correct = (preds == labels_indices) & (labels_indices == class_idx)
            correct_per_class[class_idx] += class_correct.sum().item()
            total_per_class[class_idx] += (labels_indices == class_idx).sum().item()

        # Calculate the number of correct predictions
        correct_predictions = (preds == labels_indices).sum().item()

        # Calculate accuracy
        accuracy = correct_predictions / (inputs.size(0) * 7)  # Average per batch

        pred_plate_type_idx = torch.clamp(torch.sign(output_plate_type), min=0)

        no_correct_preds_type = (pred_plate_type_idx.T.long() == plate_type).sum().item()

        # Aggregate losses
        loss = classes_loss + 4 * plate_type_loss

        # Backward pass and optimization if it's a training step
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy
        total_no_correct_preds_type += no_correct_preds_type
        total_samples += inputs.shape[0]

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    avg_acc_type = total_no_correct_preds_type / total_samples

    # Calculate class accuracies and average class accuracy
    class_accuracies = correct_per_class / total_per_class
    ###### THIS HERE IS VERY IMPORTANT. DECOMENT THIS LINE TO SEE CLASSES ACCs
    print('Classes ACCs: ', {i:class_accuracies[i].item() for i in range(num_classes)})
    avg_class_acc = class_accuracies[total_per_class > 0].mean().item()

    return avg_loss, avg_acc, avg_acc_type, avg_class_acc


def train_loop(model, 
               train_loader, 
               optimizer, 
               loss, 
               loss_fn_plate_type,
               epochs, 
               test_loader=None, 
               device="cpu", 
               folder_path=None, 
               file_name=None, 
               print_frequency=1):
    """
    Train loop functionality, for iterating, saving and (optional) loading pretrained model.
    """
    train_losses = []
    test_losses = []
    
    best_loss = torch.inf
    model = model.to(device)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(folder_path, file_name)
                )["state_dict"]
            )
        except:
            print("Couldn't load model")
    
    for e in range(1, epochs + 1):
        train_loss, train_acc, train_avg_acc_type, train_avg_class_acc = iterate(model, train_loader, optimizer, loss, loss_fn_plate_type, device=device)
        train_losses.append(train_loss)
        
        if test_loader:
            with torch.no_grad():
                test_loss, test_acc, test_avg_acc_type, test_avg_class_acc = iterate(model, test_loader, optimizer, loss, loss_fn_plate_type, is_training=False, device=device)
            test_losses.append(test_loss)
        else:
            test_loss = None
        
        #loss = test_loss if test_loader else train_loss

        if train_loss < best_loss:

            print(f"Loss improved from {best_loss} to {train_loss}. Overwriting...")
            best_loss = train_loss
    
            checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(folder_path, file_name))
        
        if e % print_frequency == 0:
            print(f"Epoch {e}/{epochs}: train_loss={train_loss} test_loss={test_loss}")
            print(f"Epoch {e}/{epochs}: train_acc={train_acc * 100}% test_acc={test_acc * 100}%")
            print(f"Epoch {e}/{epochs}: train_acc_type={train_avg_acc_type * 100}% test_acc_type={test_avg_acc_type * 100}%")
            print(f"Epoch {e}/{epochs}: train_avg_class_acc={train_avg_class_acc * 100}% test_avg_class_acc={test_avg_class_acc * 100}%")

    return train_losses, test_losses



def iterate_individual_chars(model, 
            dataloader, 
            optimizer, 
            loss_fn, 
            loss_fn_plate_type,
            is_training=True, 
            device='cuda'):
    """
    Single-iteration function, either for training or for testing.

    :param model: nn.Module, neural network
    :param dataloader: torch.utils.data.DataLoader object to iterate over
    :param optimizer: torch.optim object, used to update the model if is_training=True
    :param loss_fn: cost function
    :param is_training: bool, whether this iteration should compute gradients and update the model
    """
    # Set the model to training mode if it's a training step, otherwise to evaluation mode
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0
    total_acc = 0
    total_no_correct_preds_type = 0
    total_samples = 0

    num_classes = 36  # Number of character classes (26 letters + 10 digits)
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)

    for inputs, labels, labels_string, plate_type in tqdm(dataloader):
        if isinstance(inputs, list):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device)
        else:
            inputs = inputs.to(device)
        
        if isinstance(labels, list):
            for i in range(len(labels)):
                labels[i] = labels[i].to(device)
        else:
            labels = labels.to(device)

        if isinstance(plate_type, list):
            for i in range(len(plate_type)):
                plate_type[i] = plate_type[i].to(device)
        else:
            plate_type = plate_type.to(device)
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # VERY IMPORTANT: decomment here to get the weighted loss fucntion
        # comment to go back to the unweighed version
        #class_weights = compute_class_weights(labels, num_classes, device)
        #loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # Forward pass
        output_classes, output_plate_type = model(inputs)

        # Calculate loss if it's a training step
        classes_loss = 0
        for i in range(7):  # Loop over the sequence length
            classes_loss += loss_fn(output_classes[:, i, :], labels[:, i, :])

        classes_loss /= 7  # Average the loss over the sequence length

        plate_type_loss = loss_fn_plate_type(output_plate_type, plate_type.unsqueeze(1).float())

        # Compute accuracy
        _, preds = torch.max(output_classes, dim=2)  # preds shape will be [batch_size, number_of_classified_objects]
        labels_indices = labels.argmax(dim=2)

        # Calculate the number of correct predictions for each class
        for class_idx in range(num_classes):
            class_correct = (preds == labels_indices) & (labels_indices == class_idx)
            correct_per_class[class_idx] += class_correct.sum().item()
            total_per_class[class_idx] += (labels_indices == class_idx).sum().item()

        # Calculate the number of correct predictions
        correct_predictions = (preds == labels_indices).sum().item()

        # Calculate accuracy
        accuracy = correct_predictions / (inputs.size(0) * 7)  # Average per batch

        pred_plate_type_idx = torch.clamp(torch.sign(output_plate_type), min=0)

        no_correct_preds_type = (pred_plate_type_idx.T.long() == plate_type).sum().item()

        # Aggregate losses
        loss = classes_loss + 4 * plate_type_loss

        # Backward pass and optimization if it's a training step
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy
        total_no_correct_preds_type += no_correct_preds_type
        total_samples += inputs.shape[0]

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    avg_acc_type = total_no_correct_preds_type / total_samples

    # Calculate class accuracies and average class accuracy
    class_accuracies = correct_per_class / total_per_class
    ###### THIS HERE IS VERY IMPORTANT. DECOMENT THIS LINE TO SEE CLASSES ACCs
    print('Classes ACCs: ', {i:class_accuracies[i].item() for i in range(num_classes)})
    avg_class_acc = class_accuracies[total_per_class > 0].mean().item()

    return avg_loss, avg_acc, avg_acc_type, avg_class_acc


def train_loop_individual_chars(model, 
               train_loader, 
               optimizer, 
               loss, 
               loss_fn_plate_type,
               epochs, 
               test_loader=None, 
               device="cpu", 
               folder_path=None, 
               file_name=None, 
               print_frequency=1):
    """
    Train loop functionality, for iterating, saving and (optional) loading pretrained model.
    """
    train_losses = []
    test_losses = []
    
    best_loss = torch.inf
    model = model.to(device)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(folder_path, file_name)
                )["state_dict"]
            )
        except:
            print("Couldn't load model")
    
    for e in range(1, epochs + 1):
        train_loss, train_acc, train_avg_acc_type, train_avg_class_acc = iterate(model, train_loader, optimizer, loss, loss_fn_plate_type, device=device)
        train_losses.append(train_loss)
        
        if test_loader:
            with torch.no_grad():
                test_loss, test_acc, test_avg_acc_type, test_avg_class_acc = iterate(model, test_loader, optimizer, loss, loss_fn_plate_type, is_training=False, device=device)
            test_losses.append(test_loss)
        else:
            test_loss = None
        
        #loss = test_loss if test_loader else train_loss

        if train_loss < best_loss:

            print(f"Loss improved from {best_loss} to {train_loss}. Overwriting...")
            best_loss = train_loss
    
            checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(folder_path, file_name))
        
        if e % print_frequency == 0:
            print(f"Epoch {e}/{epochs}: train_loss={train_loss} test_loss={test_loss}")
            print(f"Epoch {e}/{epochs}: train_acc={train_acc * 100}% test_acc={test_acc * 100}%")
            print(f"Epoch {e}/{epochs}: train_acc_type={train_avg_acc_type * 100}% test_acc_type={test_avg_acc_type * 100}%")
            print(f"Epoch {e}/{epochs}: train_avg_class_acc={train_avg_class_acc * 100}% test_avg_class_acc={test_avg_class_acc * 100}%")

    return train_losses, test_losses



def test_loop(model, 
               loss, 
               loss_fn_plate_type,
               optimizer,
               test_loader=None, 
               device="cpu", 
               folder_path=None, 
               file_name=None, 
               print_frequency=1):
    """
    Train loop functionality, for iterating, saving and (optional) loading pretrained model.
    """
    train_losses = []
    test_losses = []
    
    best_loss = torch.inf
    model = model.to(device)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(folder_path, file_name)
                )["state_dict"]
            )
        except:
            print("Couldn't load model")
    
        
    if test_loader:
        with torch.no_grad():
            test_loss, test_acc, test_avg_acc_type, test_avg_class_acc = iterate(model, test_loader, optimizer, loss, loss_fn_plate_type, is_training=False, device=device)
        test_losses.append(test_loss)
    else:
        test_loss = None

    
    print(f"Epoch : test_acc={test_acc * 100}%")
    print(f"Epoch : test_acc_type={test_avg_acc_type * 100}%")
    print(f"Epoch : test_avg_class_acc={test_avg_class_acc * 100}%")

    return train_losses, test_losses