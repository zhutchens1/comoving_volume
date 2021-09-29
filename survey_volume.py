from astropy.cosmology import LambdaCDM
import numpy as np

def comoving_volume(ra1,ra2,dec1,dec2,z1,z2,H0,Om0,Ode0):
    """
    Parameters
    -----------------
    ra1 : scalar
        Lower RA in decimal degrees.
    ra2 : scalar
        Upper RA in decimal degrees.
    dec1 : scalar
        Lower declination in decimal degrees.
    dec2 : scalar
        Upper declination in decimal degrees.
    z1 : scalar
        Inner redshift.
    z2 : scalar
        Outer redshift.
    H0 : scalar
        Hubble constant for LambdaCDM cosmology.
    Om0 : scalar
        Density of non-relativistic matter in units of critical
        density at z=0.
    Ode0 : scalar
        Density of dark matter at z=0 in units of critical density.

    Returns
    ------------------
    volume : scalar
        Volume of survey spanned by input coordinates.
        Units of Mpc^3.
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    dtor=np.pi/180.
    if ra2>=ra1:
        delta_ra_rad = (ra2-ra1)*dtor
    else:
        delta_ra_rad = ((360.-ra1)+ra2)*dtor
    delta_dec_rad = np.sin(dec2*dtor)-np.sin(dec1*dtor)
    solidangle = delta_ra_rad * delta_dec_rad
    dv = cosmo.comoving_volume(z2).value - cosmo.comoving_volume(z1).value
    return (solidangle/(4*np.pi)) * dv


def comoving_volume_per_skyarea(z1,z2,H0,Om0,Ode0):
    """
    Compute the comoving volume per steradian of a spherical
    shell with inner redshift `z1` and outer radius `z2`. To
    determine the comoving volume associated with some solid
    angle A on sky, compute A*comoving_volume_per_skyarea(*).
    This function allows for generalization of the `comoving_
    volume` to surveys that do not carve out lat/lon rectang-
    les on sky.

    Parameters
    -----------------
    z1 : scalar
        Inner redshift.
    z2 : scalar
        Outer redshift.
    H0 : scalar
        Hubble constant for LambdaCDM cosmology.
    Om0 : scalar
        Density of non-relativistic matter in units of critical
        density at z=0.
    Ode0 : scalar
        Density of dark matter at z=0 in units of critical density.

    Returns
    ------------------
    volume : scalar
        Volume of shell spanned by the two input redshifts, with
        units of (Mpc)^3/sr.
    """
    cosmo = LambdaCDM(H0,Om0,Ode0)
    dv = cosmo.comoving_volume(z2).value - cosmo.comoving_volume(z1).value
    return (1/(4*np.pi)) * dv


