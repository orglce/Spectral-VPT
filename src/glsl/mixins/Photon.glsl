// #part /glsl/mixins/Photon

struct Photon {
    vec3 position;
    vec3 direction;
    vec3 transmittance;
    vec3 radiance;
    float wavelength;
    float wavelengthIntensity;
    uint bounces;
    uint samples;
};
