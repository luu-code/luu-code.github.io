---
layout: article
title: rknpu矩阵乘法使用
key: 100019
tags: 嵌入式 Linux NPU
category: blog
date: 2026-01-30 00:00:00 +08:00
mermaid: true
---

3 代码实现
3-1 关键步骤
1. 在NPU上运行矩阵乘法运算使用的是Matmul API，需要导入头文件：
  - #include "rknn_matmul_api.h"
2. 配置io参数：
<!--more-->
```C++
rknn_matmul_info matmul_info;
matmul_info.M = M;   // A矩阵的行数
matmul_info.K = K;   // A矩阵的列数
matmul_info.N = N;   // B矩阵的列数
matmul_info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;  // 输入和输出矩阵的数据类型
matmul_info.B_layout = RKNN_MM_LAYOUT_NATIVE; // B矩阵的排列方式
matmul_info.AC_layout = RKNN_MM_LAYOUT_NATIVE;// AC矩阵的排列方式
matmul_info.iommu_domain_id = 0;
// matmul_info.AC_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;  // AC矩阵的量化方式
// matmul_info.B_quant_type = RKNN_QUANT_TYPE_PER_CHANNEL_SYM; // B矩阵的量化方式
// matmul_info.group_size;  // 量化方式如果选择了Group量化才需要设置
// matmul_info.reserved;    // 不需要进行配置
```
  - M、K、N有对齐要求16或者32具体看手册
  - type输入输出数据类型 ：
    - KNN_FLOAT16_MM_FLOAT16_TO_FLOAT32：表示矩阵A和B是float16类型，矩阵C是float32类型；
    - RKNN_INT8_MM_INT8_TO_INT32：表示矩阵A和B是int8类型，矩阵C是int32类型；
    - RKNN_INT8_MM_INT8_TO_INT8：表示矩阵A、B和C是int8类型；
    - RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16：表示矩阵A、B和C是float16类型；
    - RKNN_FLOAT16_MM_INT8_TO_FLOAT32：表示矩阵A是float16类型，矩阵B是int8类型，矩阵C是float32类型；
    - RKNN_FLOAT16_MM_INT8_TO_FLOAT16：表示矩阵A是float16类型，矩阵B是int8类型，矩阵C是float16类型；
    - RKNN_FLOAT16_MM_INT4_TO_FLOAT32：表示矩阵A是float16类型，矩阵B是int4类型，矩阵C是float32类型；
    - RKNN_FLOAT16_MM_INT4_TO_FLOAT16：表示矩阵A是float16类型，矩阵B是int4类型，矩阵C是float16类型；
    - RKNN_INT8_MM_INT8_TO_FLOAT32：表示矩阵A和B是int8类型，矩阵C是float32类型;
    - RKNN_INT4_MM_INT4_TO_INT16：表示矩阵A和B是int4类型，矩阵C是int16类型；
    - RKNN_INT8_MM_INT4_TO_INT32：表示矩阵A是int8类型，B是int4类型，矩阵C是int32类型
  - B_layout和AC_layout：
    - RKNN_MM_LAYOUT_NORM：表示矩阵A和C按照原始形状排列；
    - RKNN_MM_LAYOUT_NATIVE：表示矩阵A和C按照高性能形状排列；如果使用高性能布局，需要使用接口修改矩阵布局形式
  - 量化参数配置：根据实际的数据量化方式进行配置，如果使用KNN_FLOAT16_MM_FLOAT16_TO_FLOAT32和RKNN_INT8_MM_INT8_TO_INT32两种类型不用进行配置
  - iommu_domain_id、reserved：一般不需要配置，iommu_domain_id置为0，reserved默认
3. 生成io属性：
rknn_matmul_io_attr io_attr;
memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));
4. 生成ctx
rknn_matmul_ctx ctx;
ret = rknn_matmul_create(&ctx, &matmul_info, &io_attr);
if (ret < 0)
{
    printf("rknn_matmul_create fail! ret = %d", ret);
    return -1;
}
5. 选择使用的npu核：rknn_matmul_set_core_mask(ctx, (rknn_core_mask)0);
6. 配置量化参数，前面对矩阵进行了量化才需要配置：
// rknn_quant_params params; 
// rknn_matmul_set_quant_params(ctx, &params);
7. 创建A、B、C矩阵

// Create A、B、C矩阵内存
`rknn_tensor_mem *A = rknn_create_mem(ctx, io_attr.A.size);`
if (A == NULL)
{
    printf("rknn_create_mem fail! A\n");
    return -1;
}
rknn_tensor_mem *B = rknn_create_mem(ctx, io_attr.B.size);
if (B == NULL)
{
    printf("rknn_create_mem fail! B\n");
    return -1;
}
rknn_tensor_mem *C = rknn_create_mem(ctx, io_attr.C.size);
if (C == NULL)
{
    printf("rknn_create_mem fail! C\n");
    return -1;
}

// 准备需要计算的矩阵A、B数据，和最后存放结果的矩阵C容器
void *A_Matrix = vand_a;
void *B_Matrix = vand_b;
void *B_Matrix = ret_c;
/** 将准备的矩阵A、B数据拷贝给A->virt_addr， B->virt_addr
  * 如果设置了高性能模式，需要在这里修改A、B矩阵的布局 ，然后再拷贝给B->virt_addr
  * 如果是普通布局，直接像下面这样拷贝即可
  */
// memcpy(B->virt_addr, B_Matrix, K * N * 2);

ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
if (ret < 0)
{
    printf("rknn_matmul_set_io_mem fail! A ret=%d\n", ret);
    return -1;
}
ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
if (ret < 0)
{
    printf("rknn_matmul_set_io_mem fail! B ret=%d\n", ret);
    return -1;
}
ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
if (ret < 0)
{
    printf("rknn_matmul_set_io_mem fail! C ret=%d\n", ret);
    return -1;
}
8. run：
ret = rknn_matmul_run(ctx);
if (ret < 0)
{
    printf("rknn_matmul_run error %d\n", ret);
    return -1;
}
9. 获取结果
float *npu_res_ptr = (float *)C->virt_addr;
// 如果设置的高性能布局，还需要进行格式转换
10. 释放资源
rknn_destroy_mem(ctx, A);
rknn_destroy_mem(ctx, B);
rknn_destroy_mem(ctx, C);
rknn_matmul_destroy(ctx);