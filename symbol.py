import mxnet as mx
import math
# This file contains two functions.
# The first one computes the loss with respect to YOLO loss
# The second obtain the resnet neural network


def YOLO_loss(predict,label):
    # pred (7,7,5) range:(-1,1) label: (7,7,5), range: (0,1)
    # Reshape input to desired shape (7,7,5) -> (1,49,5)
    predict = mx.sym.reshape(predict,shape=(-1,49,5))
    # transform the prediction into the range from 0 to 1
    predict_shift = (predict+1)/2
    # reshape the label
    label = mx.sym.reshape(label,shape=(-1,49,5))
    
    # split the tensor into the order of (prob,x,y,w,h)
    cl, xl, yl, wl, hl = mx.sym.split(label,num_outputs = 5, axis=2)
    cp, xp, yp, wp, hp = mx.sym.split(predict_shift,num_outputs=5,axis=2)
    
    # Differ the weights with respect to different items
    lambda_coord = math.sqrt(5.0)
    lambda_obj = 1
    lambda_noobj = math.sqrt(0.5)
    mask = cl*lambda_obj + (1-cl)*lambda_noobj
    
    # Compute the loss
    lossc = mx.sym.LinearRegressionOutput(label=cl*mask,data=cp*mask)
    lossx = mx.sym.LinearRegressionOutput(label=xl*cl*lambda_coord,data=xp*cl*lambda_coord)
    lossy = mx.sym.LinearRegressionOutput(label=yl*cl*lambda_coord,data=yp*cl*lambda_coord)
    lossw = mx.sym.LinearRegressionOutput(label = mx.sym.sqrt(wl)*cl*lambda_coord,data=mx.sym.sqrt(wp)*cl*lambda_coord)
    lossh = mx.sym.LinearRegressionOutput(label = mx.sym.sqrt(hl)*cl*lambda_coord,data=mx.sym.sqrt(hp)*cl*lambda_coord)
    
    # return the total loss
    loss = lossc + lossx + lossy +lossw + lossh
    
    return loss


# # Visualize the pretrained imagenet model
# def visualize_pretrained_model(model_path,epoch):
#     # load symbol and actual weights
#     sym,args_params, aux_params= mx.model.load_checkpoint(model_path,epoch)
#     # visualize the network
#     return mx.viz.plot_network(sym)
#     # extract the last bn layer (Figure out why we should do this)
    
# Obtain the pretrained imagenet model
def get_resnet_model(model_path,epoch,layer):
    '''
    This function inputs three arguments
    The argument - layer specifies which layer we would like to extract.
    '''
    # define the label of the output
    label = mx.sym.Variable("softmax_label")
    # load symbol and actual weights
    sym, args, aux = mx.model.load_checkpoint(model_path,epoch)
    # extract specified layer
    sym = sym.get_internals()[layer]
    # append two layers
    # THe following is the relu layer (activation layer)
    sym = mx.sym.Activation(data=sym,act_type="relu")
    # Convolution layer
    sym = mx.sym.Convolution(data=sym,kernel=(3,3),
                             num_filter=5,pad=(1,1),
                             stride=(1,1),no_bias=True) # no_bias = True since we have already consider the batch normalization
    # The output is (# of img in batch,5,7,7)
    # First normalization, get softsign. It 
    sym = sym / (1 + mx.sym.abs(sym))
    logit = mx.sym.transpose(sym,axes=(0,2,3,1),name="logit")
    # apply loss
    loss_ = YOLO_loss(logit,label)
    # mxnet special requirements
    loss = mx.sym.MakeLoss(loss_)
    
    # multi-output logit should be blocked from generating gradient
    out = mx.sym.Group([mx.sym.BlockGrad(logit),loss])
    return out
    
    