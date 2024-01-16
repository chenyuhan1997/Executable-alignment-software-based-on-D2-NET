import cv2
def resize_keep_aspectratio(image_src,dst_size):
  src_h,src_w = image_src.shape[:2]
  dst_h,dst_w = dst_size 
  
  #判断应该按哪个边做等比缩放
  h = dst_w * (float(src_h)/src_w)#按照ｗ做等比缩放
  w = dst_h * (float(src_w)/src_h)#按照h做等比缩放
  
  h = int(h)
  w = int(w)
  
  if h <= dst_h:
    image_dst = cv2.resize(image_src,(dst_w,int(h)))
  else:
    image_dst = cv2.resize(image_src,(int(w),dst_h))
  
  h_,w_ = image_dst.shape[:2]
  
  top = int((dst_h - h_) / 2)
  down = int((dst_h - h_+1) / 2)
  left = int((dst_w - w_) / 2)
  right = int((dst_w - w_+1) / 2)
  
  value = [0,0,0]
  borderType = cv2.BORDER_CONSTANT
  image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)
 
  return image_dst