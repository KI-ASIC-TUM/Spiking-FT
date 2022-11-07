#!/usr/bin/env python3
"""
Pre-processing pipeline class, with Window and Offset removal 
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algms.remove_offset
import pyrads.algms.scale
import pyrads.algms.window
import pyrads.pipeline


class PreprocPipeline(pyrads.pipeline.Pipeline):
    """
    Parent class for radar algorithms
    """
    def __init__(self, in_shape):
        alg_chain = self.generate_alg_chain(in_shape)
        super().__init__(alg_chain)


    def generate_alg_chain(self, in_shape):
        # Define algorithms parameters
        window_params = {
            "axis": -1,
            "window_type": "hann"
        }
        scale_params = {
            "mode": "max"
        }
        remove_offset_alg = pyrads.algms.remove_offset.RemoveOffset(
            in_shape
        )
        window_alg = pyrads.algms.window.Window(
            remove_offset_alg.out_data_shape,
            **window_params
        )
        scale_alg = pyrads.algms.scale.Scale(
            window_alg.out_data_shape,
            **scale_params
        )

        # Create list of defined algorithms
        algorithms = [
            remove_offset_alg,
            window_alg,
            scale_alg,
        ]
        return algorithms
