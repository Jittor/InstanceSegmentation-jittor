import jittor as jt

CUDA_HEADER = r'''
#include<cfloat>
using namespace std;
'''

CUDA_SRC=r'''
__global__ void MaskProbForward(
    const float* embed_pixel,
    const float* embed_center,
    const float* sigma_center,
    const int* boxes,
    const int* box_areas,
    const int area_sum, 
    const int num_pixel,
    const int mask_width,
    const int dim,
    float* probs) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= area_sum)
    return;

  int center_id = 0;
  int cur_area_sum = box_areas[0];
  int last_area_sum = 0;
  while(i >= cur_area_sum){
    center_id+=1;
    last_area_sum = cur_area_sum;
    cur_area_sum += box_areas[center_id];
  }
  int pixel_in_id = i - last_area_sum;

  const int* cur_box = &boxes[4*center_id];
  int box_width = cur_box[2] - cur_box[0];
  int x = pixel_in_id % box_width + cur_box[0];
  int y = pixel_in_id / box_width + cur_box[1];
  int pixel_id = y * mask_width + x;

  const float* p_ep = embed_pixel + pixel_id*dim;
  const float* p_ec = embed_center + center_id*dim;
  float norm2 = 0.0;

  for (int d = 0; d < dim; ++d){
    norm2 = norm2 + (*p_ep - *p_ec) * (*p_ep - *p_ec);
    p_ep++;
    p_ec++;
  }

  float p = expf(-norm2*sigma_center[center_id]);
  probs[center_id*num_pixel+pixel_id] = p;
} 


memset(out_p,0,out->size);


auto dim = in0_shape1;
auto area_sum = in5->loop_options.data().at("area_sum");
auto mask_width = in5->loop_options.data().at("mask_width");
int num_pixel = in0_shape0;
int b = area_sum/512;
if(area_sum%512!=0){
    b++;
}


dim3 blocks((long)b);
dim3 threads(512);

MaskProbForward<<<blocks, threads>>>(
      in0_p,
      in1_p,
      in2_p,
      in3_p,
      in4_p,
      area_sum,
      num_pixel,
      mask_width,
      dim,
      out_p
    );

'''

def mask_prob_cuda(embed_pixel,embed_center,sigma_center,boxes,box_areas,area_sum,mask_width):
    assert embed_pixel.ndim== 2, "embed_pixel should be MxDim"
    assert embed_center.ndim == 2, "embed_center should be NxDim"
    assert sigma_center.ndim == 1, "sigma_center should be N"
    assert embed_pixel.shape[1] == embed_center.shape[1], "Dim should the same"
    assert embed_center.shape[0] == sigma_center.shape[0], "center number should be the same"
    assert embed_center.shape[0] == boxes.shape[0], "center number and box number should be the same"
    
    output_shape = (embed_pixel.shape[0],embed_center.shape[0])
    if output_shape[0]*output_shape[1]==0:
        return jt.array([],embed_pixel.dtype)
    output_type = embed_pixel.dtype
    option = jt.empty((0,))
    option.compile_options = {"area_sum": int(area_sum), "mask_width":int(mask_width) }
    inputs = [embed_pixel,embed_center,sigma_center,boxes,box_areas, option]
    output = jt.code(output_shape,output_type,inputs, cuda_header=CUDA_HEADER,cuda_src=CUDA_SRC)
    return output