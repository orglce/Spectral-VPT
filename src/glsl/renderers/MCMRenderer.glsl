// #part /glsl/shaders/renderers/MCM/integrate/vertex

#version 300 es

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

out vec2 vPosition;

void main() {
    vec2 position = vertices[gl_VertexID];
    vPosition = position;
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/MCM/integrate/fragment

#version 300 es
precision mediump float;
precision mediump sampler2D;
precision mediump sampler3D;

#define EPS 1e-5

// #link /glsl/mixins/Photon
@Photon
// #link /glsl/mixins/intersectCube
@intersectCube

@constants
@random/hash/pcg
@random/hash/squashlinear
@random/distribution/uniformdivision
@random/distribution/square
@random/distribution/disk
@random/distribution/sphere
@random/distribution/exponential
@rgbaToFloat

@unprojectRand
@unproject

uniform sampler2D uPosition;
uniform sampler2D uDirection;
uniform sampler2D uTransmittance;
uniform sampler2D uRadiance;
uniform sampler2D uWavelength;

uniform sampler3D uVolume;
uniform sampler2D uTransferFunction;
uniform sampler2D uEnvironment;

uniform mat4 uMvpInverseMatrix;
uniform vec2 uInverseResolution;
uniform float uRandSeed;
uniform float uBlur;

uniform float uExtinction;
uniform float uAnisotropy;
uniform uint uMaxBounces;
uniform uint uSteps;

in vec2 vPosition;

layout (location = 0) out vec4 oPosition;
layout (location = 1) out vec4 oDirection;
layout (location = 2) out vec4 oTransmittance;
layout (location = 3) out vec4 oRadiance;
layout (location = 4) out vec4 oWavelength;

vec3 wavelengthToRGB(float wavelengthClamped) {
    float factor;
    float red, green, blue;
    float wavelength = 380.0 + wavelengthClamped * (780.0 - 380.0);

    if ((wavelength >= 380.0) && (wavelength < 440.0)) {
        red = -(wavelength - 440.0) / (440.0 - 380.0);
        green = 0.0;
        blue = 1.0;
    } else if ((wavelength >= 440.0) && (wavelength < 490.0)) {
        red = 0.0;
        green = (wavelength - 440.0) / (490.0 - 440.0);
        blue = 1.0;
    } else if ((wavelength >= 490.0) && (wavelength < 510.0)) {
        red = 0.0;
        green = 1.0;
        blue = -(wavelength - 510.0) / (510.0 - 490.0);
    } else if ((wavelength >= 510.0) && (wavelength < 580.0)) {
        red = (wavelength - 510.0) / (580.0 - 510.0);
        green = 1.0;
        blue = 0.0;
    } else if ((wavelength >= 580.0) && (wavelength < 645.0)) {
        red = 1.0;
        green = -(wavelength - 645.0) / (645.0 - 580.0);
        blue = 0.0;
    } else if ((wavelength >= 645.0) && (wavelength < 781.0)) {
        red = 1.0;
        green = 0.0;
        blue = 0.0;
    } else {
        red = 0.0;
        green = 0.0;
        blue = 0.0;
    }

    // Let the intensity fall off near the vision limits

    if ((wavelength >= 380.0) && (wavelength < 420.0)) {
        factor = 0.3 + 0.7 * (wavelength - 380.0) / (420.0 - 380.0);
    } else if ((wavelength >= 420.0) && (wavelength < 701.0)) {
        factor = 1.0;
    } else if ((wavelength >= 701.0) && (wavelength < 781.0)) {
        factor = 0.3 + 0.7 * (780.0 - wavelength) / (780.0 - 700.0);
    } else {
        factor = 0.0;
    }

    float intensityMax = 255.0;
    float gamma = 2.2;

    vec3 rgb;

    // Don't want 0^x = 1 for x <> 0
    rgb.r = (red == 0.0) ? 0.0 : pow(red * factor, gamma) * intensityMax;
    rgb.g = (green == 0.0) ? 0.0 : pow(green * factor, gamma) * intensityMax;
    rgb.b = (blue == 0.0) ? 0.0 : pow(blue * factor, gamma) * intensityMax;

    return rgb;
}

void resetPhoton(inout uint state, inout Photon photon) {
    vec3 from, to;
    unprojectRand(state, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);
    photon.wavelength = random_uniform(state);
    photon.wavelengthIntensity = 1.0;
    photon.direction = normalize(to - from);
    photon.bounces = 0u;
    vec2 tbounds = max(intersectCube(from, photon.direction), 0.0);
    photon.position = from + tbounds.x * photon.direction;
    photon.transmittance = vec3(1);
}

vec4 sampleEnvironmentMap(vec3 d) {
    vec2 texCoord = vec2(atan(d.x, -d.z), asin(-d.y) * 2.0) * INVPI * 0.5 + 0.5;
    return texture(uEnvironment, texCoord);
}

vec4 sampleVolumeColor(vec3 position) {
    vec2 volumeSample = texture(uVolume, position).rg;
    vec4 transferSample = texture(uTransferFunction, volumeSample);
    return transferSample;
}

float sampleHenyeyGreensteinAngleCosine(inout uint state, float g) {
    float g2 = g * g;
    float c = (1.0 - g2) / (1.0 - g + 2.0 * g * random_uniform(state));
    return (1.0 + g2 - c * c) / (2.0 * g);
}

vec3 sampleHenyeyGreenstein(inout uint state, float g, vec3 direction) {
    // generate random direction and adjust it so that the angle is HG-sampled
    vec3 u = random_sphere(state);
    if (abs(g) < EPS) {
        return u;
    }
    float hgcos = sampleHenyeyGreensteinAngleCosine(state, g);
    float lambda = hgcos - dot(direction, u);
    return normalize(u + lambda * direction);
}

float max3(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float mean3(vec3 v) {
    return dot(v, vec3(1.0 / 3.0));
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 gradient(vec3 pos, float h) {
    vec3 positive = vec3(
    sampleVolumeColor(pos + vec3(h, 0, 0)).a,
    sampleVolumeColor(pos + vec3(0, h, 0)).a,
    sampleVolumeColor(pos + vec3(0, 0, h)).a
    );
    vec3 negative = vec3(
    sampleVolumeColor(pos - vec3(h, 0, 0)).a,
    sampleVolumeColor(pos - vec3(0, h, 0)).a,
    sampleVolumeColor(pos - vec3(0, 0, h)).a
    );
    return (positive - negative) / (2.0 * h);
}

vec3 gradientRefractionIndex(vec3 pos, float h) {
    vec3 positive = vec3(
    rgb2hsv(sampleVolumeColor(pos + vec3(h, 0, 0)).rgb).x,
    rgb2hsv(sampleVolumeColor(pos + vec3(0, h, 0)).rgb).x,
    rgb2hsv(sampleVolumeColor(pos + vec3(0, 0, h)).rgb).x
    );
    vec3 negative = vec3(
    rgb2hsv(sampleVolumeColor(pos - vec3(h, 0, 0)).rgb).x,
    rgb2hsv(sampleVolumeColor(pos - vec3(0, h, 0)).rgb).x,
    rgb2hsv(sampleVolumeColor(pos - vec3(0, 0, h)).rgb).x
    );
    return (positive - negative) / (2.0 * h);
}

vec3 refractDirection(vec3 incidentDir, vec3 gradient) {
    vec3 normal = gradient;
    if (dot(incidentDir, normal) > 0.0) {
        normal = -normal;
    }
    float etaRatio = length(gradient);
    if (dot(incidentDir, gradient) > 0.0) {
        etaRatio = 1.0 / etaRatio;
    }

    float cosI = abs(dot(normal, incidentDir));
    float sinT2 = etaRatio * etaRatio * (1.0 - cosI * cosI);

    if (sinT2 > 1.0) {
//         Total internal reflection
                return vec3(0.0); // Return a zero vector or any other appropriate value
    }

    float cosT = sqrt(1.0 - sinT2);
    return etaRatio * incidentDir + (etaRatio * cosI - cosT) * normal;
}

float getTransmittanceSpectrum(vec3 rgb, float wavelength) {
//    change transmittance according to the material
//    implement reflectance spectrum
//    rgb represents a material in this case
    return wavelength;
}

float getAbsorbtionSpectrum(vec3 rgb, float wavelength) {
//    change absorbtion according to the material
//    implement reflectance spectrum
//    rgb represents a material in this case
    return rgb2hsv(rgb).x;
}

void main() {
    Photon photon;
    vec2 mappedPosition = vPosition * 0.5 + 0.5;
    photon.position = texture(uPosition, mappedPosition).xyz;

    vec4 directionAndBounces = texture(uDirection, mappedPosition);
    photon.direction = directionAndBounces.xyz;
    photon.bounces = uint(directionAndBounces.w + 0.5);

    vec4 transmittance = texture(uTransmittance, mappedPosition);
    photon.transmittance = transmittance.rgb;

    vec4 radianceAndSamples = texture(uRadiance, mappedPosition);
    photon.radiance = radianceAndSamples.rgb;
    photon.samples = uint(radianceAndSamples.w + 0.5);

    vec4 wavlength = texture(uWavelength, mappedPosition);
    photon.wavelength = wavlength.x;
    photon.wavelengthIntensity = wavlength.y;

    uint state = hash(uvec3(floatBitsToUint(mappedPosition.x), floatBitsToUint(mappedPosition.y), floatBitsToUint(uRandSeed)));
    for (uint i = 0u; i < uSteps; i++) {
        float dist = random_exponential(state, uExtinction);
        photon.position += dist * photon.direction;

        vec3 normal = normalize(gradient(photon.position, 0.005));
        float lambert = max(dot(normal, -photon.direction), 0.0);
        vec4 volumeSample = sampleVolumeColor(photon.position);
//        float refraction_index = rgbaToFloat(volumeSample);
        float refraction_index = rgb2hsv(volumeSample.rgb).x;

        float PNull = 1.0 - volumeSample.a;
        float PScattering;
        if (photon.bounces >= uMaxBounces) {
            PScattering = 0.0;
        } else {
            PScattering = volumeSample.a * max3(volumeSample.rgb);
        }
        float PAbsorption = 1.0 - PNull - PScattering;

        float fortuneWheel = random_uniform(state);
        if (any(greaterThan(photon.position, vec3(1))) || any(lessThan(photon.position, vec3(0)))) {
            // out of bounds
            vec4 envSample = sampleEnvironmentMap(photon.direction);
            vec3 radiance = photon.wavelengthIntensity * wavelengthToRGB(photon.wavelength) * envSample.rgb;
            photon.samples++;
            photon.radiance += (radiance - photon.radiance) / float(photon.samples);
            resetPhoton(state, photon);
        } else if (fortuneWheel < PAbsorption) {
            // absorption
            vec3 radiance = vec3(0);
            photon.samples++;
            photon.wavelengthIntensity *= getAbsorbtionSpectrum(volumeSample.rgb, photon.wavelength);
            photon.radiance -= photon.radiance * photon.wavelengthIntensity / float(photon.samples);

            resetPhoton(state, photon);
        } else if (fortuneWheel < PAbsorption + PScattering) {
            // scattering
            photon.transmittance *= volumeSample.rgb;
            photon.wavelengthIntensity  *= getTransmittanceSpectrum(volumeSample.rgb, photon.wavelength);
//            photon.direction = sampleHenyeyGreenstein(state, uAnisotropy, photon.direction);
//            photon.direction = refractDirection(photon.direction, gradientRefractionIndex(photon.position, 0.005));
            photon.bounces++;
        } else {
            // null collision
        }
    }

    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, float(photon.wavelength));
    oRadiance = vec4(photon.radiance, float(photon.samples));
    oWavelength = vec4(photon.wavelength, photon.wavelengthIntensity, 0, 0);
}

// #part /glsl/shaders/renderers/MCM/render/vertex

#version 300 es

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

out vec2 vPosition;

void main() {
    vec2 position = vertices[gl_VertexID];
    vPosition = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/MCM/render/fragment

#version 300 es
precision mediump float;
precision mediump sampler2D;

uniform sampler2D uColor;

in vec2 vPosition;

out vec4 oColor;

void main() {
    oColor = vec4(texture(uColor, vPosition).rgb, 1);
}

// #part /glsl/shaders/renderers/MCM/reset/vertex

#version 300 es

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

out vec2 vPosition;

void main() {
    vec2 position = vertices[gl_VertexID];
    vPosition = position;
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/MCM/reset/fragment

#version 300 es
precision mediump float;

// #link /glsl/mixins/Photon
@Photon
// #link /glsl/mixins/intersectCube
@intersectCube

@constants
@random/hash/pcg
@random/hash/squashlinear
@random/distribution/uniformdivision
@random/distribution/square
@random/distribution/disk
@random/distribution/sphere
@random/distribution/exponential

@unprojectRand
@unproject

uniform mat4 uMvpInverseMatrix;
uniform vec2 uInverseResolution;
uniform float uRandSeed;
uniform float uBlur;

in vec2 vPosition;

layout (location = 0) out vec4 oPosition;
layout (location = 1) out vec4 oDirection;
layout (location = 2) out vec4 oTransmittance;
layout (location = 3) out vec4 oRadiance;
layout (location = 4) out vec4 oWavelength;

void main() {
    Photon photon;
    vec3 from, to;
    uint state = hash(uvec3(floatBitsToUint(vPosition.x), floatBitsToUint(vPosition.y), floatBitsToUint(uRandSeed)));
    unprojectRand(state, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);

    photon.direction = normalize(to - from);
    vec2 tbounds = max(intersectCube(from, photon.direction), 0.0);
    photon.position = from + tbounds.x * photon.direction;
    photon.transmittance = vec3(1);
    photon.radiance = vec3(1);
    photon.bounces = 0u;
    photon.samples = 0u;
    photon.wavelength = random_uniform(state);
    photon.wavelengthIntensity = 1.0;
    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, float(photon.wavelength));
    oRadiance = vec4(photon.radiance, float(photon.samples));
    oWavelength = vec4(photon.wavelength, photon.wavelengthIntensity, 0, 0);
}
