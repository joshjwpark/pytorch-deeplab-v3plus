# Python includes
import timeit
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# from torchvision.utils import make_grid

# Custom includes
import pascal
import utils
import DeepLabV3Plus
import custom_transforms as tr


# -------------------------------------------------------------------------------
# Initial Parameters
# -------------------------------------------------------------------------------
gpu_id = 0
print('Using GPU: {}'.format(gpu_id))
nEpochs = 20
resume_epoch = 4
nValInterval = 3            # Run on validation set every nTestInterval epochs
valBatch = 6                # Validation batch size
useVal = True               # Test on validation data
snapshot = 1                # Store a model every snapshot epochs
backbone = 'resnet101'      # backbone of DeepLabV3Plus model


# -------------------------------------------------------------------------------
# Include in Report
# -------------------------------------------------------------------------------
p = OrderedDict()           # Parameters to include in report
p['trainBatch'] = 6         # Training batch size
p['nAveGrad'] = 1           # Average the gradient of several iterations
p['lr'] = 1e-7              # Learning rate
p['wd'] = 5e-4              # Weight decay
p['momentum'] = 0.9         # Momentum
p['epoch_size'] = 10        # How many epochs to change learning rate


# -------------------------------------------------------------------------------
# Load/Save Directories
# -------------------------------------------------------------------------------
save_directory_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_directory_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_directory_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

# save_directory = os.path.join(save_directory_root, 'run', 'run_' + str(run_id))
# save_directory = os.path.join(save_directory_root, str(run_id))
save_directory = save_directory_root


# -------------------------------------------------------------------------------
# Load and Initialize Data
# -------------------------------------------------------------------------------
transforms_train = transforms.Compose([
    tr.FixedResize(size=(512, 512)),
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor(),
    ])
transforms_val = transforms.Compose([
    tr.FixedResize(size=(512, 512)),
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor(),
    ])

voc_train = pascal.VOCSegmentation(split='train', transform=transforms_train)
voc_val = pascal.VOCSegmentation(split='val', transform=transforms_val)

trainloader = DataLoader(voc_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
valloader = DataLoader(voc_val, batch_size=valBatch, shuffle=False, num_workers=0)


# -------------------------------------------------------------------------------
# Initialize Network
# -------------------------------------------------------------------------------
net = DeepLabV3Plus.DeepLabV3_Plus(nInputChannels=3, n_classes=21, os=16, pretrained=True)

modelname = 'DeepLabV3Plus-' + backbone + '-voc'
criterion = utils.cross_entropy2d
# Select optimizer
optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
p['optimizer'] = str(optimizer)

if gpu_id >= 0:
    torch.cuda.device(device=gpu_id)
    net.cuda()

# utils.generate_param_report(os.path.join(save_directory, exp_name + '.txt'), p)

num_img_train = len(trainloader)
num_img_val = len(valloader)
running_loss_train = 0.0
running_loss_val = 0.0
aveGrad = 0
global_step = 0


# -------------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------------
if resume_epoch == 0:
    print("Training DeepLabV3+ from scratch...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_directory, 'models', modelname + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_directory, 'models', modelname + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))  # Load all tensors onto the CPU

if resume_epoch != nEpochs:
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        print('Training...')

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            learning_rate = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', learning_rate)
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=p['momentum'], weight_decay=p['wd'])

        net.train()
        for ii, sample_batched in enumerate(trainloader):
            inputs, labels = sample_batched['image'], sample_batched['label']

            # -----------------
            # Forward Pass
            # -----------------
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)

            loss = criterion(outputs, labels, size_average=False, batch_average=True)
            running_loss_train += loss.item()

            # Print loss
            if ii % num_img_train == (num_img_train - 1):
                running_loss_train = running_loss_train / num_img_train
                # writer.add_scalar('data/total_loss_epoch', running_loss_train, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_train)
                running_loss_train = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # -----------------
            # Backward Pass
            # -----------------
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # -----------------
            # Update weights
            # -----------------
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

        # Save model
        if (epoch % snapshot) == snapshot -1:
            torch.save(net.state_dict(), os.path.join(save_directory, 'models', modelname + '_epoch-' + str(epoch) + '.pth'))
            print("Saved model at {}\n".format(os.path.join(save_directory, 'models', modelname + '_epoch-' + str(epoch) + '.pth')))

        # ------------
        # Validate
        # ------------
        if useVal and epoch % nValInterval == (nValInterval - 1):
            print('Validating...')
            total_iou = 0.0
            net.eval()
            for ii, sample_batched in enumerate(valloader):
                inputs, labels = sample_batched['image'], sample_batched['label']

                # -----------------
                # Forward Pass
                # -----------------
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                if gpu_id >= 0:
                    inputs, labels = inputs.cuda(), labels.cuda()

                with torch.no_grad():
                    outputs = net.forward(inputs)

                predictions = torch.max(outputs, 1)[1]

                loss = criterion(outputs, labels, size_average=False, batch_average=True)
                running_loss_val += loss.item()

                total_iou += utils.get_iou(predictions, labels)

                # Print loss and MIoU
                if ii % num_img_val == num_img_val - 1:
                    miou = total_iou / (ii*valBatch + inputs.data.shape[0])
                    running_loss_val = running_loss_val / num_img_val

                    print('Validation')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * valBatch + inputs.data.shape[0]))
                    print('Loss: %f' % running_loss_val)
                    print('MIoU: %f\n' % miou)
                    running_loss_val = 0
