from typing import TYPE_CHECKING, Dict
from qcengine.programs.model import ProgramHarness
from qcelemental.util import safe_version, which_import
from qcelemental.models import AtomicResult, Provenance
from qcengine.exceptions import InputError

if TYPE_CHECKING:
    from qcelemental.models import AtomicInput
    from qcengine.config import TaskConfig


class QCMLForgeHarness(ProgramHarness):
    """A harness to run QCMLForge models <https://github.com/awallace3/qcmlforge>"""

    _CACHE = {}

    _defaults = {
        "name": "qcmlforge",
        "scratch": False,
        "thread_safe": True,
        "thread_parallel": False,
        "node_parallel": False,
        "managed_memory": False,
    }

    version_cache: Dict[str, str] = {}

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        return which_import(
            "qcmlforge",
            return_bool=True,
            raise_error=raise_error,
            raise_msg="Please install via `pip install git+https://github.com/awallace3/qcmlforge.git@main`",
        )

    def get_version(self) -> str:
        self.found(raise_error=True)

        which_prog = which_import("apnet_pt")
        if which_prog not in self.version_cache:
            import qcmlforge

            self.version_cache[which_prog] = safe_version(qcmlforge.__version__)

        return self.version_cache[which_prog]

    def load_model(self, name: str):
        model_name = name.lower()
        if model_name in self._CACHE:
            return self._CACHE[model_name]

        from apnet_pt import load_model

        model = load_model(model_name=model_name)
        self._CACHE[model_name] = model
        return self._CACHE[model_name]

    def compute(self, input_data: "AtomicInput", config: "TaskConfig"):
        self.found(raise_error=True)
        import torch
        from qcengine.units import ureg

        # check we can run on the set of elements
        known_elements = {"H", "B", "C", "N", "O", "F", "Na", "P", "S", "Cl", "Br"}
        target_elements = set(input_data.molecule.symbols)

        unknown_elements = target_elements - known_elements
        if unknown_elements:
            raise InputError(f"QCMLForge model {input_data.model.method} does not support elements {unknown_elements}.")

        method = input_data.model.method
        # load the model using the method as the file name
        model = self.load_model(name=method)

        # build the required input data
        aimnet_input = {
            "coord": torch.tensor(
                [input_data.molecule.geometry * ureg.conversion_factor("bohr", "angstrom")],
                dtype=torch.float64,
                device="cpu",
            ),
            "numbers": torch.tensor([input_data.molecule.atomic_numbers], dtype=torch.long, device="cpu"),
            "charge": torch.tensor([input_data.molecule.molecular_charge], dtype=torch.float64, device="cpu"),
        }

        if input_data.driver == "gradient":
            aimnet_input["coord"].requires_grad_(True)
        out = model(aimnet_input)

        ret_data = {
            "success": False,
            "properties": {
                "return_energy": out["energy"].item() * ureg.conversion_factor("eV", "hartree"),
                "return_gradient": (
                    -1.0 * out["forces"][0].detach().numpy() * ureg.conversion_factor("eV / angstrom", "hartree / bohr")
                ),
                "calcinfo_natom": len(input_data.molecule.atomic_numbers),
            },
            "extras": input_data.extras.copy(),
        }
        # update with calculated extras
        ret_data["extras"]["qcmlforge"] = {
            "charges": out["charges"].detach()[0].cpu().numpy(),
            "ensemble_charges_std": out["charges_std"].detach()[0].cpu().numpy(),
            "ensemble_energy_std": out["energy_std"].item(),
            "ensemble_forces_std": out["forces_std"].detach()[0].cpu().numpy(),
        }
        if input_data.driver == "energy":
            ret_data["return_result"] = ret_data["properties"]["return_energy"]
        elif input_data.driver == "gradient":
            ret_data["return_result"] = ret_data["properties"]["return_gradient"]
        else:
            raise InputError(
                f"QCMLForge can only compute energy and gradients driver methods. Requested {input_data.driver} not supported."
            )

        ret_data["provenance"] = Provenance(creator="qcmlforge", version=self.get_version(), routine="load_model")

        ret_data["success"] = True

        return AtomicResult(**{**input_data.dict(), **ret_data})
