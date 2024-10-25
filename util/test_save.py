import torch 
from util.load_data import *
import os

def record(task_name, epoch, val_file, model, size):
    
    #model = model.to("cpu")
    model.eval()
    current_directory = os.getcwd()
    folder_name = current_directory + '/val_' + task_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    epoch_name = folder_name + '/' + str(epoch)
    if not os.path.exists(epoch_name):
        os.makedirs(epoch_name)
    
    address = val_file#"MSRS-main/train"
    
    names = get_address(address)
    img_type = get_image_type(names[0],address+"/vi/")
    
    for name in names:
        img_vi = cv2.imread(address+'/vi/'+ name + img_type)
        img_ir = cv2.imread(address+'/ir/'+ name + img_type)
        img_vi_ycrcb = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
        img_vi = cv2.cvtColor(img_vi_ycrcb[:,:,0], cv2.COLOR_GRAY2BGR)
        #img_vi = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
        img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)
        h,w,_ = img_vi.shape
        target_size = (w, h)
        
        img_vi = cv2.resize(img_vi,(size,size),cv2.INTER_CUBIC) # type: ignore
        img_ir = cv2.resize(img_ir,(size,size),cv2.INTER_CUBIC) # type: ignore
        #img_com = 255*(img_ir / 255)**(0.7)

        img_vi = normalize_image(img_vi,size)
        img_ir = normalize_image(img_ir,size)
        #img_com = normalize_image(img_com,size)

        #model.eval()
        vi_in = prepare_image(img_vi).to("cuda:0")
        ir_in = prepare_image(img_ir).to("cuda:0")
        #com_in = prepare_image(img_com).to("cuda:2")
        
        result = model.module.val_forward(vi_in, ir_in)
        ###mask = model.module.val_forward(vi_in, ir_in)[1]

        pred = model.module.unpatchify(result) #EFD
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()
        
        #、、、、pred_mask = model.module.unpatchify(mask)
        ####pred_mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        assert pred[0].shape[2] == 3
        pred = torch.clip((pred[0] * imagenet_std + imagenet_mean) * 255, 0, 255)
        
        # assert pred_mask[0].shape[2] == 3
        # pred_mask = torch.clip((pred_mask[0] * imagenet_std + imagenet_mean) * 255, 0, 255)

        path = epoch_name +"/"+name+img_type
        path_gray = epoch_name +"/"+"gray_"+name+img_type
        # path_mask = epoch_name +"/"+"mask_"+name+img_type

        # pred_mask = pred_mask.numpy()
        # pred_mask = cv2.convertScaleAbs(pred_mask)
        # y_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2GRAY)
        # y_mask = cv2.resize(y_mask,target_size,interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(path_mask,y_mask)


        pred = pred.numpy()
        pred = cv2.convertScaleAbs(pred)
        y = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(path_gray,y)
        img_vi_ycrcb = cv2.resize(img_vi_ycrcb,(size,size),cv2.INTER_CUBIC) # type: ignore
        img_vi_ycrcb[:,:,0] = y
        out1 = cv2.cvtColor(img_vi_ycrcb, cv2.COLOR_YCrCb2BGR)
        out = cv2.resize(out1,target_size,interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path,out)
        #print(path)
    
    