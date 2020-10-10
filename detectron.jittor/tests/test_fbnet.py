# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import numpy as np
import jittor as jt
import detectron.modeling.backbone.fbnet_builder as fbnet_builder



TEST_CUDA = True


def _test_primitive(self, op_name, op_func, N, C_in, C_out, expand, stride):
    op = op_func(C_in, C_out, expand, stride)
    input = jt.random([N, C_in, 7, 7]).float32()
    output = op(input)
    self.assertEqual(
        output.shape[:2], ([N, C_out]),
        'Primitive {} failed for shape {}.'.format(op_name, input.shape)
    )


class TestFBNetBuilder(unittest.TestCase):
    def test_identity(self):
        id_op = fbnet_builder.Identity(20, 20, 1)
        input = jt.random([10, 20, 7, 7]).float32()
        output = id_op(input)
        np.testing.assert_array_equal(input.numpy(), output.numpy())

        id_op = fbnet_builder.Identity(20, 40, 2)
        input = jt.random([10, 20, 7, 7]).float32()
        output = id_op(input)
        np.testing.assert_array_equal(output.shape, [10, 40, 4, 4])

    def test_primitives(self):
        ''' Make sures the primitives runs '''
        jt.flags.use_cuda=0
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            _test_primitive(
                self,
                op_name, op_func,
                N=20, C_in=16, C_out=32, expand=4, stride=1
            )

    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_primitives_cuda(self):
        ''' Make sures the primitives runs on cuda '''
        jt.flags.use_cuda=1
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            _test_primitive(
                self,
                op_name, op_func,
                N=20, C_in=16, C_out=32, expand=4, stride=1
            )

    def test_primitives_empty_batch(self):
        ''' Make sures the primitives runs '''
        jt.flags.use_cuda=0
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            # test empty batch size
            _test_primitive(
                self,
                op_name, op_func,
                N=0, C_in=16, C_out=32, expand=4, stride=1
            )

    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_primitives_cuda_empty_batch(self):
        jt.flags.use_cuda=1
        ''' Make sures the primitives runs '''
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            # test empty batch size
            _test_primitive(
                self,
                op_name, op_func,
                N=0, C_in=16, C_out=32, expand=4, stride=1
            )

if __name__ == "__main__":
    unittest.main()
