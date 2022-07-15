import numpy as np
import torch
import torch.optim as optim
from torch_geometric.datasets import WikipediaNetwork
from models import RWGAT
from box import Box
import yaml
import os, argparse
from shutil import copyfile
from names_generator import generate_name

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_config', '-c', help="path to the yaml config file", type=str, default='config.yml')

# load the config file
config = Box.from_yaml(filename='config.yml', Loader=yaml.FullLoader)

# initialize the datasets and dataloaders
dataset = WikipediaNetwork(root='./data/', name='chameleon', geom_gcn_preprocess=True)
data = dataset[0]

if __name__ == "__main__":

    # create model saving dir and copy config file to run dir
    run_name = generate_name()
    current_run_dir = os.path.join(config.io_settings.run_dir, run_name)
    os.makedirs(os.path.join(current_run_dir, 'trained_models'))
    copyfile(parser.parse_args().yaml_config, os.path.join(current_run_dir, 'config.yml'))

    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')

    # initialize the model
    model = RWGAT(num_node_features=dataset.num_node_features, num_classes=dataset.num_classes, **config.model_settings)

    # if using a pretrained model, load it here
    if config.io_settings.pretrained_model:
        model.load_state_dict(torch.load(config.io_settings.pretrained_model, map_location=torch.device('cpu')))

    # send the model and data to the gpu if available
    model.to(device)
    data.to(device)

    # save the intial state
    torch.save(model.state_dict(), os.path.join(current_run_dir, 'init_model.pt'))

    test_accs = []

    # loop over the different splits of the data
    for d_i in range(data.train_mask.shape[1]):
        # load the initial state
        model.load_state_dict(torch.load(os.path.join(current_run_dir, 'init_model.pt'), map_location=torch.device('cpu')))

        # define optimizer
        optimizer = optim.Adam(model.parameters(), lr=float(config.hyperparameters.start_lr))

        # define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.hyperparameters.lr_decay_gamma)

        # training loop
        print('Starting run split {} on {}'.format(d_i, next(model.parameters()).device))
        for epoch in range(config.hyperparameters.epochs):

            model.train()
            # reset the gradients back to zero
            optimizer.zero_grad()

            # run forward pass and compute the batch training loss
            out = model(data.x, data.edge_index)
            train_loss = model.compute_loss(out[data.train_mask[:, d_i]], data.y[data.train_mask[:, d_i]]) / data.train_mask[:, d_i].sum()
            train_acc = (out[data.train_mask[:, d_i]].argmax(dim=1) == data.y[data.train_mask[:, d_i]]).sum() / data.train_mask[:, d_i].sum()

            # perform batched SGD parameter update
            train_loss.backward()
            optimizer.step()

            # step the scheduler
            scheduler.step()

            # compute validation loss
            model.eval()
            with torch.no_grad():
                # compute the batch validation loss
                out = model(data.x, data.edge_index)
                validation_loss = model.compute_loss(out[data.val_mask[:, d_i]], data.y[data.val_mask[:, d_i]]) / data.val_mask[:, d_i].sum()
                validation_acc = (out[data.val_mask[:, d_i]].argmax(dim=1) == data.y[data.val_mask[:, d_i]]).sum() / data.val_mask[:, d_i].sum()

            # save the model with the best validation loss
            if epoch == 0:
                best_validation_loss = validation_loss
            else:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save(model.state_dict(), os.path.join(current_run_dir, 'trained_models', 'best_{}.pt'.format(d_i)))

        # run the best model on the test set
        model.load_state_dict(torch.load(os.path.join(current_run_dir, 'trained_models', 'best_{}.pt'.format(d_i)),map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            # compute the test acc
            out = model(data.x, data.edge_index)
            test_acc = (out[data.test_mask[:, d_i]].argmax(dim=1) == data.y[data.test_mask[:, d_i]]).sum() / data.test_mask[:, d_i].sum()
            test_accs.append(test_acc.cpu().numpy().item())
        print('Test Acc: {}'.format(test_acc))

    print('----- End of 10 runs -----')
    print('Average test accuracy: {}'.format(np.mean(test_accs)))
    print('All test accuracies: ')
    print(test_accs)