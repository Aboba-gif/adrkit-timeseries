"""Распределённый источник при заданных коэффициентах ADR.

JVP/VJP вычисляются по источнику при фиксированных коэффициентах уравнения.
"""
from dataclasses import dataclass
import json

import numpy as np

from adrkit.backends.bounded import BoundedSolver
from adrkit.state_maps import forcing_state_map
from adrkit.coefficients import ADRUnits, ReactionLaw, ScalarField, VectorField
from adrkit.spaces import ArraySpace


@dataclass(frozen=True)
class Diffusivity:
    """Аффинное поле диффузии по координате и времени.

    Parameters
    ----------
    offset : float, optional
        Постоянная часть диффузии в единицах м²/с примера.
    """
    offset: float = .04

    def evaluate(self, points, time):
        """Вычислить коэффициент диффузии в точках.

        Parameters
        ----------
        points : ndarray, shape (n, 2)
            Координаты в метрах.
        time : float
            Физическое время в секундах.

        Returns
        -------
        diffusion : ndarray, shape (n,)
            Коэффициент диффузии в м²/с; функция аффинна по первой
            координате и времени.
        """
        return self.offset+.015*points[:, 0]+.01*time


def main():
    """Решить ADR с заданными полями и распределённым источником."""
    units = ADRUnits(length='m', time='s', concentration='kg/m^3')
    reaction = ReactionLaw(
        value=lambda c, xy, t: .05*c**3,
        derivative=lambda c, xy, t: .15*c**2,
        pure=True, unit=units.reaction, derivative_unit=units.loss)
    model = dict(
        horizon=.2, time_origin=0., units=units,
        diffusion=ScalarField(Diffusivity(), unit=units.diffusion),
        velocity=VectorField(lambda xy, t: np.column_stack((.1+.02*xy[:, 1],
                                                            np.zeros(len(xy)))),
                             unit=units.velocity, time_dependent=False),
        linear_loss=ScalarField(lambda xy, t: .1+.02*t, unit=units.loss),
        reaction=reaction, diffusion_form='divergence', advection_form='advective',
        source_position=None)
    solver = BoundedSolver(model, interior_points=15, time_steps=24,
                           domain=(0., 1., 0., 1.))
    # Размерность источника: концентрация / время. Строка t=0 не используется:
    # начальное состояние нулевое; правая часть берётся на правых концах шагов.
    x, y = solver.xy.T
    pulse = np.exp(-((solver.times-.07)/.04)**2)
    forcing = pulse[:, None]*(np.sin(np.pi*x)*np.sin(np.pi*y))[None, :]
    trajectory = solver.solve_forcing(forcing)

    coordinates = (tuple(solver.times), tuple(map(tuple, solver.xy)))
    source_space = ArraySpace(forcing.shape, axes=('time', 'node'),
                              coordinates=coordinates, units=units.reaction)
    state_space = ArraySpace(forcing.shape, axes=('time', 'node'),
                             coordinates=coordinates, units=units.concentration)
    mapping = forcing_state_map(solver, domain=source_space, codomain=state_space)
    random = np.random.default_rng(23)
    direction, cotangent = random.normal(size=forcing.shape), random.normal(size=forcing.shape)
    jvp = mapping.jvp.jvp(forcing, direction)
    vjp = mapping.vjp.vjp(forcing, cotangent)
    left, right = np.vdot(jvp, cotangent), np.vdot(direction, vjp)
    eps = 1e-5
    finite_difference = (mapping.predict.predict(forcing+eps*direction)
                         -mapping.predict.predict(forcing-eps*direction))/(2*eps)
    report = dict(
        purpose='Расчёт с заданными коэффициентами; производные по аддитивному источнику',
        state_shape=list(trajectory.states.shape),
        final_max_concentration=float(np.max(trajectory.states[-1])),
        max_scaled_residual=trajectory.max_scaled_residual,
        jvp_max_absolute_difference=float(np.max(np.abs(jvp-finite_difference))),
        adjoint_absolute_difference=float(abs(left-right)),
        coefficient_derivatives_implemented=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
