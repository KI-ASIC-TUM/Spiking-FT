# Standard libraries
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import typing as ty
# Third-party libraries
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.nc.type import LavaNcType
try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:
    class NetL2:
        pass
from lava.magma.core.model.nc.tables import Nodes
from lava.magma.core.resources import NeuroCore, Loihi2NeuroCore
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.nc.model import AbstractNcProcessModel
from lava.magma.core.model.nc.var import NcVar
from lava.proc.lif.process import LIF, AbstractLIF
from lava.magma.core.learning.constants import W_TRACE_FRACTIONAL_PART
from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
from lava.proc.io.source import RingBuffer as SpikeGenerator
from lava.proc.io.sink import RingBuffer as Sink
from lava.proc.dense.process import Dense
from lava.magma.core.process.process import LogConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.process.variable import Var
from lava.magma.core.callback_fx import NxSdkCallbackFx
from lava.utils.profiler import Profiler
# Local libraries
import pyrads.algorithm


class Probes(NxSdkCallbackFx):
    def __init__(self):
        self.trace_probes = []
        self.syn_probes = []
        self.u_probes = []
        self.v_probes = []
        self.spikes_probe = []

    def pre_run_callback(self, board, var_id_to_model_map):
        nxChip = board.nxChips[0]
        nxCorePost = nxChip.nxCores[0]
        mon = board.monitor
        self.u_probes.append(
            mon.probe(
                nxCorePost.neuronInterface.group[0].cxState,
                [20],
                "u"
            )[0]
        )
        self.v_probes.append(
            mon.probe(
                nxCorePost.neuronInterface.group[0].cxState,
                [20],
                "v"
            )[0]
        )

    def post_run_callback(self, board, var_id_to_model_map):
        np.save("probe_u.npy", self.u_probes[0].data)
        np.save("probe_v.npy", self.v_probes[0].data)
        # print("LIF-post u", self.u_probes[0].data)
        # print("LIF-post v", self.v_probes[0].data)


class TCBS(AbstractLIF):
    """Model based on Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        t_half: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        charging_bias: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            charging_bias=charging_bias,
            t_half=t_half,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.t_half = Var(shape=(1,), init=t_half)
        self.charging_bias = Var(shape=(1,), init=charging_bias)
        self.vth = Var(shape=(1,), init=vth)


@implements(proc=TCBS, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
@tag("ucoded")
class TCBSModel(AbstractNcProcessModel):
    """Implementation of a Temporal charge-before-spike (TCBS) neural process
    model that defines the behavior of micro-coded (ucoded) LIF neurons on
    Loihi 2. In its current form, this process model exactly matches the
    hardcoded LIF behavior with one exception: To improve performance by a
    factor of two, the negative saturation behavior of v is switched off.
    """

    # Declare port implementation
    a_in: NcInPort = LavaNcType(NcInPort, np.int16, precision=16)
    s_out: NcOutPort = LavaNcType(NcOutPort, np.int32, precision=24)
    # Declare variable implementation
    u: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    v: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    du: NcVar = LavaNcType(NcVar, np.int16, precision=12)
    dv: NcVar = LavaNcType(NcVar, np.int16, precision=12)
    charging_bias = LavaNcType(NcVar, np.int16, precision=12)
    t_half = LavaNcType(NcVar, np.int16, precision=12)
    bias_mant: NcVar = LavaNcType(NcVar, np.int16, precision=13)
    bias_exp: NcVar = LavaNcType(NcVar, np.int16, precision=3)
    vth: NcVar = LavaNcType(NcVar, np.int32, precision=17)

    def allocate(self, net: NetL2):
        """Allocates neural resources in 'virtual' neuro core."""

        shape = np.product(list(self.proc_params["shape"]))

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        ucode_file = os.path.join(curr_dir, "tcbs.dasm")

        # vth_reg = np.left_shift(self.vth.var.init, 6)
        vth_reg = self.vth.var.init

        # Allocate neurons
        neurons_cfg: Nodes = net.neurons_cfg.allocate_ucode(
            shape=(1,),
            ucode=ucode_file,
            vth=vth_reg,
            t_half=self.t_half,
            charging_bias=self.charging_bias,
            du=4096 - self.du.var.get(),
            dv=4096 - self.dv.var.get(),
        )
        neurons: Nodes = net.neurons.allocate_ucode(
            shape=shape,
            u=self.u,
            v=self.v,
            bias=self.bias_mant.var.get() * 2 ** (self.bias_exp.var.get()),
        )

        # Allocate output axons
        ax_out: Nodes = net.ax_out.allocate(shape=shape, num_message_bits=0)
        # Connect InPort of Process to neurons
        self.a_in.connect(neurons)
        # Connect Nodes
        neurons.connect(neurons_cfg)
        neurons.connect(ax_out)
        # Connect output axon to OutPort of Process
        ax_out.connect(self.s_out)


class SDFT(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "S-DFT"

    def __init__(self, *args, **kwargs):
        # Load DFT parameters
        self.n_dims = kwargs.get("n_dims")
        # Load simulations parameters
        self.timesteps = kwargs.get("timesteps", 100)
        self.alpha = kwargs.get("alpha", 0.4)
        self.time_step = 1
        self.out_type = kwargs.get("out_type", "spike")
        self.out_abs = kwargs.get("out_abs", False)
        self.normalize = kwargs.get("normalize", False)
        self.off_bins = kwargs.get("off_bins", 0)
        self.debug = kwargs.get("debug", False)
        # Load parent class
        super().__init__(*args, **kwargs)
        if self.out_type not in ["spike", "voltage"]:
            raise ValueError(
                    "{} not a valid value for 'out_type'".format(self.out_type)
            )
        # SNN population variables
        self.spikes = np.zeros(self.layer_dim)
        self.voltage = np.zeros((2*self.timesteps,) + self.layer_dim)
        # Initialize SNN and its connections
        self.calculate_weights()
        v_th = self.alpha*0.25*self.weights_re[0].sum()*self.timesteps
        # Charge-and-spike neuron parameters
        self.neuron_params = {}
        self.neuron_params["threshold"] = v_th
        self.neuron_params["t_silent"] = self.timesteps
        self.neuron_params["i_offset"] = 2*v_th // self.timesteps
        # TODO: move here self.init_snn()

    def calculate_out_shape(self):
        """
        Split in imag and real components. Only half spectrum is useful
        """
        self.out_data_shape = self.in_data_shape[:-1]
        self.out_data_shape += (int(self.in_data_shape[-1]/2)-self.off_bins, )
        if not self.out_abs:
            self.out_data_shape += (2, )
        self.layer_dim = self.in_data_shape[:-1] + (int(self.in_data_shape[-1]/2), 2)

    def calculate_weights(self):
        """
        Calculate the weights based on the DFT equation
        """
        c = 2 * np.pi/self.in_data_shape[-1]
        n = np.arange(self.in_data_shape[-1]).reshape(self.in_data_shape[-1], 1)
        k = np.arange(self.in_data_shape[-1]).reshape(1, self.in_data_shape[-1])
        trig_factors = np.dot(n, k) * c
        real_weights = np.cos(trig_factors)[:self.layer_dim[-2]]
        imag_weights = -np.sin(trig_factors)[:self.layer_dim[-2]]
        # Normalize for the allowed range in SpiNNaker
        self.weights_re = real_weights*127
        self.weights_im = imag_weights*127
        self.weights = np.vstack((self.weights_re, self.weights_im))
        self.weights = self.weights.astype(np.int8)

    def init_snn(self, spike_in):
        """
        Initializes compartments depending on the number of samples
        """
        # Create neuron populations' instances
        self.sg = SpikeGenerator(data=spike_in)
        self.py2nx = PyToNxAdapter(shape=(self.in_data_shape[-1], ))
        self.dense1 = Dense(weights=self.weights)
        self.tcbs = TCBS(
            shape=(self.in_data_shape[-1],),
            t_half=self.neuron_params["t_silent"],
            vth=self.neuron_params["threshold"],
            charging_bias=self.neuron_params["i_offset"]
        )
        self.dense2 = Dense(weights=np.eye(self.in_data_shape[-1]))
        self.nx2py = NxToPyAdapter(shape=(self.in_data_shape[-1], ))
        self.sink = Sink(
            shape=(self.in_data_shape[-1], ),
            buffer=2*self.timesteps
        )

        # Interconnect populations
        self.sg.s_out.connect(self.py2nx.inp)
        self.py2nx.out.connect(self.dense1.s_in)
        self.dense1.a_out.connect(self.tcbs.a_in)
        self.tcbs.s_out.connect(self.dense2.s_in)
        self.dense2.s_in.connect(self.nx2py.inp)
        self.nx2py.out.connect(self.sink.a_in)
        return

    def simulate(self, spike_times):
        """
        Run the S-DFT over the total simulation time
        """
        # if self.debug:
        probes = Probes()
        run_config = Loihi2HwCfg(callback_fxs=[probes],)
        self.tcbs.run(
            condition=RunSteps(num_steps=2*self.timesteps),
            run_cfg=run_config
        )
        self.spikes = self.sink.data.get()
        self.voltages = self.tcbs.v.get()
        self.tcbs.stop()
        return

    def format_input(self, in_data):
        """
        Convert the input to a binary spike train per input neuron
        """
        spike_array = np.zeros((self.in_data_shape[-1], 2*self.timesteps))
        for i in range(self.in_data_shape[-1]):
            spike_array[i][in_data[i]] = 1
        return spike_array

    def format_output(self, spikes):
        """
        Tranform output dictionary into output data array
        """
        spike_times =  np.argmax(spikes, axis=1)
        # All neurons that didn't spike are forced to spike in the last step,
        # since the spike-time of 1 corresponds to the lowest possible value.
        spikes_all = np.where(spike_times == 0, 2*self.timesteps, spike_times)
        reshaped_spikes = np.vstack((
            spikes_all[:self.layer_dim[-2]],
            spikes_all[self.layer_dim[-2]:]
        )).transpose()
        # Decode from TTFS spikes to input data format
        output = 1.5*self.timesteps - reshaped_spikes.reshape(self.layer_dim)
        output = output[..., self.off_bins:,:]
        if self.out_abs:
            output = np.sqrt(output[..., 0]**2 + output[..., 1]**2)
        if self.normalize:
            output = output - output.min()
            output /= output.max()
        return output


    def _run(self, in_data):
        spike_in = self.format_input(in_data)
        self.init_snn(spike_in)
        # Run SNN
        self.simulate(spike_in)
        # Fetch  and format SNN output
        output = self.format_output(self.spikes)
        return output
