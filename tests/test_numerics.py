"""Numerical regression checks against independent quadrature and SciPy references."""
import importlib.util
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import integrate, interpolate, special

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nanocmb as cmb


class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bg = cmb.compute_background(cmb.params)
        cls.thermo = cmb.compute_thermodynamics(cls.bg, cmb.params)
        cls.pgrid = cmb.setup_perturbation_grid(cls.bg, cls.thermo)

    def test_conformal_time_against_quadrature(self):
        for changes in ({}, {'h': 0.70, 'omega_c_h2': 0.13}, {'N_eff': 4.0}):
            bg = cmb.compute_background(dict(cmb.params, **changes))
            a = np.r_[0., np.geomspace(1e-9, 1, 25), 1.2]
            reference = [integrate.quad(cmb.dtauda, 0, x, args=(bg,),
                                       epsrel=1e-12, epsabs=1e-12)[0] for x in a]
            np.testing.assert_allclose(cmb.conformal_time(a, bg), reference, rtol=1e-10)

    def test_time_origin_and_early_helium_opacity(self):
        p = self.pgrid
        self.assertAlmostEqual(p['sp_a_x'][-1], self.bg['tau0'], places=9)
        self.assertAlmostEqual(cmb._cubic_eval(p['sp_a_x'], p['sp_a_c'], self.bg['tau0']), 1.)
        t = p['sp_a_x'][0]
        a = cmb._cubic_eval(p['sp_a_x'], p['sp_a_c'], t)
        opacity = cmb._cubic_eval(p['sp_op_x'], p['sp_op_c'], t)
        self.assertAlmostEqual(opacity*a*a/self.bg['akthom'], 1+2*self.bg['f_He'])

    def test_akima_columns_are_independent(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-10, 0, 50)
        xx = np.linspace(-10, 0, 300)
        y = rng.normal(size=(50, 9))*np.geomspace(1e-40, 1e6, 9)
        y[:, 0], y[:, 1], y[:, 2] = 0., 1e-20, x*1e-14
        reference = np.column_stack([interpolate.Akima1DInterpolator(x, v)(xx) for v in y.T])
        scale = np.maximum(np.max(np.abs(reference), axis=0), 1e-100)
        np.testing.assert_allclose(cmb._akima_columns(x,y,xx)/scale, reference/scale,
                                   rtol=1e-12, atol=1e-13)

    def test_small_x_bessel_derivatives(self):
        self.assertEqual(cmb._small_x_bessel(2, 0.)[2], 2/15)
        x = np.geomspace(1e-10, 0.0999, 40)
        for ell in (2, 3, 10):
            j = special.spherical_jn(ell, x)
            d = special.spherical_jn(ell, x, derivative=True)
            dd = -2*d/x + (ell*(ell+1)/x**2-1)*j
            for actual, reference in zip(cmb._small_x_bessel(ell,x), (j,d,dd)):
                np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=0.)

    def test_bessel_table_build_and_reload(self):
        orders = np.array([2,3,5])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cmb,'__file__',str(Path(directory)/'nanocmb.py')):
            cmb._bessel_cache.clear()
            x0,inv_dx,nx,j,jnext = cmb._build_bessel_tables(orders,5.,.02)
            x = x0+np.arange(nx)/inv_dx
            for i,ell in enumerate(orders):
                np.testing.assert_allclose(j[i],special.spherical_jn(ell,x),rtol=1e-10,atol=1e-15)
                np.testing.assert_allclose(jnext[i],special.spherical_jn(ell+1,x),rtol=1e-10,atol=1e-15)
            np.testing.assert_array_equal(j[1],jnext[0])
            cmb._bessel_cache.clear()
            reloaded = cmb._build_bessel_tables(orders,5.,.02)
            np.testing.assert_array_equal(reloaded[3],j)
            np.testing.assert_array_equal(reloaded[4],jnext)
            cmb._bessel_cache.clear()

    def test_los_against_direct_bessel_quadrature_without_numba(self):
        spec = importlib.util.spec_from_file_location('nanocmb_plain', cmb.__file__)
        plain = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'numba': None}):
            spec.loader.exec_module(plain)
        self.assertFalse(plain.NUMBA_AVAILABLE)
        rng = np.random.default_rng(7)
        tau = np.r_[0., np.geomspace(1e-5, 3., 80)]
        chi = 3.-tau
        ks = np.array([.01, .1, .7])
        sources = [rng.normal(size=(len(ks),len(tau))) for _ in range(4)]
        # Use a fine lookup grid to separate table error from projection errors.
        ell, dx = 2, .0001
        table_x = np.arange(0., 4., dx)
        table = special.spherical_jn(ell, table_x)
        next_table = special.spherical_jn(ell+1, table_x)
        args = (ell,ks,chi,cmb._trapezoid_weights(tau),*sources,
                0.,1/dx,len(table_x),table,next_table)
        actual = cmb._los_integrals(*args)
        np.testing.assert_allclose(plain._los_numpy(*args,chunk=2),actual,rtol=1e-12,atol=1e-14)
        x = ks[:,None]*chi
        j = special.spherical_jn(ell,x)
        d = special.spherical_jn(ell,x,derivative=True)
        dd = np.full_like(x,2/15)
        mask = x > 0
        dd[mask] = -2*d[mask]/x[mask]+(ell*(ell+1)/x[mask]**2-1)*j[mask]
        reference = (np.trapezoid(sources[0]*j+sources[1]*d+sources[2]*dd,tau,axis=1),
                     np.trapezoid(sources[3]*j,tau,axis=1))
        np.testing.assert_allclose(actual,reference,rtol=3e-6,atol=2e-7)

    def test_slip_preserves_combined_momentum(self):
        p = self.pgrid
        tau, k = 100., .001
        y = cmb.adiabatic_ics(k,tau,self.bg,p)
        args = tuple(p[key] for key in ('bg_vec','sp_a_x','sp_a_c','sp_op_x','sp_op_c','sp_cs_x','sp_cs_c'))
        dy = cmb._boltzmann_rhs(tau,y,k,*args)
        a,H,gg,gr,gc,gb,dr,dq,Z,sigma,*_ = cmb._common_terms(tau,y,k,*args[:3])
        opacity = cmb._cubic_eval(p['sp_op_x'],p['sp_op_c'],tau)
        cs2 = cmb._cubic_eval(p['sp_cs_x'],p['sp_cs_c'],tau)
        self.assertLess(k/opacity, .01)
        self.assertLess(1/(opacity*tau), .01)
        ratio = 4*gg/(3*gb)
        vb,clxb,clxg = y[cmb.IX_VB],y[cmb.IX_CLXB],y[cmb.IX_G]
        pig = 32/45*k/opacity*(sigma+vb)
        expected = -H*vb+k*cs2*clxb+ratio*k/4*(clxg-2*pig)
        np.testing.assert_allclose(dy[cmb.IX_VB]+.75*ratio*dy[cmb.IX_G+1],expected,rtol=1e-12)
        self.assertGreater(abs(dy[cmb.IX_VB]-.75*dy[cmb.IX_G+1]),1e-15)

    def test_solver_completeness_and_initial_time(self):
        incomplete = SimpleNamespace(success=True,y=np.zeros((3,1)),message='truncated fixture')
        with patch.object(cmb.integrate,'solve_ivp',return_value=incomplete):
            with self.assertRaisesRegex(RuntimeError,'RECFAST integration incomplete'):
                cmb.compute_recombination(self.bg,cmb.params)
        with patch.object(cmb.integrate,'solve_ivp',return_value=incomplete) as solve:
            with self.assertRaisesRegex(RuntimeError,'ODE solver failed/incomplete'):
                cmb.evolve_k(.5,self.bg,self.thermo,self.pgrid,np.array([1.,2.]))
            self.assertAlmostEqual(solve.call_args.args[1][0],.02)

    def test_zero_reionization_retains_residual_electrons(self):
        p = dict(cmb.params,tau_reion=0.)
        thermo = cmb.compute_thermodynamics(self.bg,p)
        self.assertFalse(thermo['reionization'])
        self.assertTrue(0 < thermo['xe'][-1] < .001)

    def test_stdin_uses_serial_fallback(self):
        class SerialProbe(Exception):
            pass
        with patch.object(sys.modules['__main__'],'__file__','<stdin>',create=True), \
                patch.object(mp,'get_start_method',return_value='spawn'), \
                patch.object(mp,'Pool') as pool, patch.object(cmb,'_boltzmann_rhs'), \
                patch.object(cmb,'evolve_k',side_effect=SerialProbe):
            with self.assertRaises(SerialProbe):
                cmb.compute_cls(self.bg,self.thermo,dict(cmb.params,ell_max=2),
                                k_arr=[1e-4,1e-3,1e-2],k_fine=[1e-4,1e-3],tau_out=[1.,2.])
            pool.assert_not_called()

    def test_multipole_endpoint_and_serial_pipeline(self):
        for maximum in (2,3,237,2500):
            self.assertEqual(cmb.ell_grid(maximum)[-1],maximum)
        np.testing.assert_array_equal(cmb.ell_grid(39),np.arange(2,40))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cmb,'__file__',str(Path(directory)/'nanocmb.py')):
            cmb._bessel_cache.clear()
            result = cmb.compute_cls(self.bg,self.thermo,dict(cmb.params,ell_max=2),
                                     n_workers=1,los_workers=1,k_arr=[1e-4,1e-3,1e-2],
                                     k_fine=np.geomspace(1e-4,1e-2,30),N_tau=100)
            cmb._bessel_cache.clear()
        np.testing.assert_array_equal(result['ells'],[2])
        self.assertTrue(all(np.isfinite(result['Dl_'+s]).all() for s in ('TT','EE','TE')))


if __name__=='__main__':
    unittest.main()
