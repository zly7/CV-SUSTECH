import numpy as np
## 这个是比较快的大矩阵乘法版本
def my_imfilter(image, filter):

  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """
  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  origin_shape = image.shape[:2]
  padding = [int((filter.shape[0]-1) // 2), int((filter.shape[1]-1) // 2)]
  image = np.pad(image, [
    (padding[0], padding[0]),
    (padding[1], padding[1]),
    (0, 0) # 在channel不会进行填充
  ])
  krnl_h = filter.shape[0]
  krnl_w = filter.shape[1]
  final_shape = (origin_shape[0], origin_shape[1], krnl_h, krnl_w)
  channel_array = []
  for i in range(image.shape[-1]):
    
    img_one_channel = (image[:,:,i]).copy() # 没有这个copy简直是世纪大坑，直接切片不会复制内存,大部分情况不会错，但是遇到stride_tricks直接出问题
    # print(f"img_one_channel shape : {img_one_channel.shape}")
    assert len(img_one_channel.shape) == 2
    
    strides = np.array([img_one_channel.shape[1], 1, img_one_channel.shape[1], 1]) * img_one_channel.itemsize

    img_neighoured = np.lib.stride_tricks.as_strided(img_one_channel, final_shape, strides)  

    # 这个numpy库做并行化处理，就是搞成一个大矩阵，可以理解成截取大矩阵的方式 
    #然后kernel复制很多遍，这个直接对应元素相乘，就会变快
    # print(img_neighoured.shape)
    filtered_image = np.sum(np.sum(img_neighoured * filter, -1), -1)
    channel_array.append(filtered_image.reshape(filtered_image.shape[0],filtered_image.shape[1],1))
    # print(filtered_image.reshape(filtered_image.shape[0],filtered_image.shape[1],1).shape)
  if len(channel_array) == 1:
    return channel_array[0]
  else:
    final_img = channel_array[0]
    for index in range(1,len(channel_array)):
      final_img = np.concatenate([final_img,channel_array[index]],axis=2) # concatenate的巨坑传入只有一个参数
    return final_img 
## 这个是比较慢的矩阵对应相乘版本
def my_imfilter2(image, filter):
  origin_shape = image.shape[:2]
  padding = [int((filter.shape[0]-1) // 2), int((filter.shape[1]-1) // 2)] # 减号没加括号,人麻了
  padded_image = np.pad(image, [
    (padding[0], padding[0]),
    (padding[1], padding[1]),
    (0, 0) # 在channel不会进行填充
  ])
  result = None
  size_x = filter.shape[0] 
  size_y = filter.shape[1]
  for c in range(image.shape[2]):
    output_array = np.zeros((origin_shape[0], 
                          origin_shape[1],
                          1)) 
    # print(output_array.shape)
    for x in range(origin_shape[0]): # -size_x + 1 is to keep the window within the bounds of the image
      for y in range(origin_shape[1]):

          # Creates the window with the same size as the filter
          # window = padded_image[x:x + size_x, y:y + size_y][c]# 这个写法没报错，人麻了
          window = padded_image[x:x + size_x, y:y + size_y,c]
          # print(f"window_shape{window.shape}")(3,3)
          # Sums over the product of the filter and the window
          output_values = np.sum(filter * window, axis=(0, 1)) 

          # Places the calculated value into the output_array
          output_array[x, y] = output_values # 这个会自动转型把float变成ndarray
          # print(output_array[x, y])
    if result is None:
      # result = output_array.reshape(output_array.shape[0],output_array.shape[1],1)
      result = output_array
    else:
      # output_array = output_array.reshape(output_array.shape[0],output_array.shape[1],1)
      result = np.concatenate([result,output_array],axis=2)
  return result
# 这个写法有问题
def my_imfilter3(image, filter):
  origin_shape = image.shape[:2]
  padding = [int((filter.shape[0]-1) // 2), int((filter.shape[1]-1) // 2)]
  padded_image = np.pad(image, [
    (padding[0], padding[0]),
    (padding[1], padding[1]),
    (0, 0) # 在channel不会进行填充
  ])

  size_x = filter.shape[0] 
  size_y = filter.shape[1]

  output_array = np.zeros((origin_shape[0], 
                        origin_shape[1],
                        image.shape[-1])) 
  for x in range(origin_shape[0]): # -size_x + 1 is to keep the window within the bounds of the image
    for y in range(origin_shape[1]):

        # Creates the window with the same size as the filter
        window = padded_image[x:x + size_x, y:y + size_y]

        # Sums over the product of the filter and the window
        output_values = np.sum(filter * window, axis=(0, 1)) # 这个搞出来是绿色，矩阵传播
        # print(output_values)

        # Places the calculated value into the output_array
        output_array[x, y] = output_values

  return output_array
    
    



  ### END OF STUDENT CODE ####
  ############################



def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###

  low_frequencies = my_imfilter(image1,filter)
  low_frequencies2 = my_imfilter(image2, filter)
  high_frequencies = image2 - low_frequencies2
  low_frequencies = low_frequencies.clip(0,1)
  high_frequencies = high_frequencies.clip(0,1)
  hybrid_image = low_frequencies + high_frequencies 
  hybrid_image = hybrid_image.clip(0,1)
  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
