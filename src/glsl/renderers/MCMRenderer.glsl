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

float D65[531] = float[] (0.0341,0.36014,0.68618,1.01222,1.33826,1.6643,1.99034,2.31638,2.64242,2.96846,3.2945,4.98865,6.6828,8.37695,10.0711,11.7652,13.4594,15.1535,16.8477,18.5418,20.236,21.9177,23.5995,25.2812,26.963,28.6447,30.3265,32.0082,33.69,35.3717,37.0535,37.343,37.6326,37.9221,38.2116,38.5011,38.7907,39.0802,39.3697,39.6593,39.9488,40.4451,40.9414,41.4377,41.934,42.4302,42.9265,43.4228,43.9191,44.4154,44.9117,45.0844,45.257,45.4297,45.6023,45.775,45.9477,46.1203,46.293,46.4656,46.6383,47.1834,47.7285,48.2735,48.8186,49.3637,49.9088,50.4539,50.9989,51.544,52.0891,51.8777,51.6664,51.455,51.2437,51.0323,50.8209,50.6096,50.3982,50.1869,49.9755,50.4428,50.91,51.3773,51.8446,52.3118,52.7791,53.2464,53.7137,54.1809,54.6482,57.4589,60.2695,63.0802,65.8909,68.7015,71.5122,74.3229,77.1336,79.9442,82.7549,83.628,84.5011,85.3742,86.2473,87.1204,87.9936,88.8667,89.7398,90.6129,91.486,91.6806,91.8752,92.0697,92.2643,92.4589,92.6535,92.8481,93.0426,93.2372,93.4318,92.7568,92.0819,91.4069,90.732,90.057,89.3821,88.7071,88.0322,87.3572,86.6823,88.5006,90.3188,92.1371,93.9554,95.7736,97.5919,99.4102,101.228,103.047,104.865,106.079,107.294,108.508,109.722,110.936,112.151,113.365,114.579,115.794,117.008,117.088,117.169,117.249,117.33,117.41,117.49,117.571,117.651,117.732,117.812,117.517,117.222,116.927,116.632,116.336,116.041,115.746,115.451,115.156,114.861,114.967,115.073,115.18,115.286,115.392,115.498,115.604,115.711,115.817,115.923,115.212,114.501,113.789,113.078,112.367,111.656,110.945,110.233,109.522,108.811,108.865,108.92,108.974,109.028,109.082,109.137,109.191,109.245,109.3,109.354,109.199,109.044,108.888,108.733,108.578,108.423,108.268,108.112,107.957,107.802,107.501,107.2,106.898,106.597,106.296,105.995,105.694,105.392,105.091,104.79,105.08,105.37,105.66,105.95,106.239,106.529,106.819,107.109,107.399,107.689,107.361,107.032,106.704,106.375,106.047,105.719,105.39,105.062,104.733,104.405,104.369,104.333,104.297,104.261,104.225,104.19,104.154,104.118,104.082,104.046,103.641,103.237,102.832,102.428,102.023,101.618,101.214,100.809,100.405,100.0,99.6334,99.2668,98.9003,98.5337,98.1671,97.8005,97.4339,97.0674,96.7008,96.3342,96.2796,96.225,96.1703,96.1157,96.0611,96.0065,95.9519,95.8972,95.8426,95.788,95.0778,94.3675,93.6573,92.947,92.2368,91.5266,90.8163,90.1061,89.3958,88.6856,88.8177,88.9497,89.0818,89.2138,89.3459,89.478,89.61,89.7421,89.8741,90.0062,89.9655,89.9248,89.8841,89.8434,89.8026,89.7619,89.7212,89.6805,89.6398,89.5991,89.4091,89.219,89.029,88.8389,88.6489,88.4589,88.2688,88.0788,87.8887,87.6987,87.2577,86.8167,86.3757,85.9347,85.4936,85.0526,84.6116,84.1706,83.7296,83.2886,83.3297,83.3707,83.4118,83.4528,83.4939,83.535,83.576,83.6171,83.6581,83.6992,83.332,82.9647,82.5975,82.2302,81.863,81.4958,81.1285,80.7613,80.394,80.0268,80.0456,80.0644,80.0831,80.1019,80.1207,80.1395,80.1583,80.177,80.1958,80.2146,80.4209,80.6272,80.8336,81.0399,81.2462,81.4525,81.6588,81.8652,82.0715,82.2778,81.8784,81.4791,81.0797,80.6804,80.281,79.8816,79.4823,79.0829,78.6836,78.2842,77.4279,76.5716,75.7153,74.859,74.0027,73.1465,72.2902,71.4339,70.5776,69.7213,69.9101,70.0989,70.2876,70.4764,70.6652,70.854,71.0428,71.2315,71.4203,71.6091,71.8831,72.1571,72.4311,72.7051,72.979,73.253,73.527,73.801,74.075,74.349,73.0745,71.8,70.5255,69.251,67.9765,66.702,65.4275,64.153,62.8785,61.604,62.4322,63.2603,64.0885,64.9166,65.7448,66.573,67.4011,68.2293,69.0574,69.8856,70.4057,70.9259,71.446,71.9662,72.4863,73.0064,73.5266,74.0467,74.5669,75.087,73.9376,72.7881,71.6387,70.4893,69.3398,68.1904,67.041,65.8916,64.7421,63.5927,61.8752,60.1578,58.4403,56.7229,55.0054,53.288,51.5705,49.8531,48.1356,46.4182,48.4569,50.4956,52.5344,54.5731,56.6118,58.6505,60.6892,62.728,64.7667,66.8054,66.4631,66.1209,65.7786,65.4364,65.0941,64.7518,64.4096,64.0673,63.7251,63.3828,63.4749,63.567,63.6592,63.7513,63.8434,63.9355,64.0276,64.1198,64.2119,64.304,63.8188,63.3336,62.8484,62.3632,61.8779,61.3927,60.9075,60.4223,59.9371,59.4519,58.7026,57.9533,57.204,56.4547,55.7054,54.9562,54.2069,53.4576,52.7083,51.959,52.5072,53.0553,53.6035,54.1516,54.6998,55.248,55.7961,56.3443,56.8924,57.4406,57.7278,58.015,58.3022,58.5894,58.8765,59.1637,59.4509,59.7381,60.0253,60.3125);

vec3 wavelengthToRGB(float wavelength) {
    float factor;
    float red, green, blue;

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

    float intensityMax = 1.0;
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
//    unprojectRand(state, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);
    unproject(vPosition, uMvpInverseMatrix, from, to);
    photon.wavelength = random_uniform(state) * (720.0 - 320.0) + 320.0;
    photon.wavelengthTransmittance = 1.0;
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

float get_gaussian_value(float x, float a, float b, float c) {
    return a * exp(-(x - b)*(x - b) / (2.0 * c * c));
}

float get_D65_spectrum_value(float wavelength) {
    return D65[int(wavelength)];
}

float get_transmission_spectrum_value(float wavelength) {
    return get_gaussian_value(wavelength, 1.0, 480.0, 20.0);
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

    vec4 wavelength = texture(uWavelength, mappedPosition);
    photon.wavelength = wavelength.x;
    photon.wavelengthRadiance = wavelength.y;
    photon.wavelengthTransmittance = wavelength.z;

    uint state = hash(uvec3(floatBitsToUint(mappedPosition.x), floatBitsToUint(mappedPosition.y), floatBitsToUint(uRandSeed)));
//    for (uint i = 0u; i < uSteps; i++) {
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
            photon.samples++;
            vec4 envSample = sampleEnvironmentMap(photon.direction);

            vec3 radiance = photon.transmittance * envSample.rgb;
            float photonRadiance = photon.wavelengthTransmittance * 1.0;

            photon.radiance += (radiance - photon.radiance) ;
            photon.wavelengthRadiance += (photonRadiance - photon.wavelengthRadiance) / float(photon.samples);

            resetPhoton(state, photon);
        } else if (fortuneWheel < PAbsorption) {
            // absorption
            photon.samples++;

            photon.radiance -= photon.radiance ;
            photon.wavelengthRadiance -= photon.wavelengthRadiance / float(photon.samples);

            resetPhoton(state, photon);
        } else if (fortuneWheel < PAbsorption + PScattering) {
            // scattering
            photon.transmittance *= volumeSample.rgb;
            photon.wavelengthTransmittance *= get_transmission_spectrum_value(photon.wavelength);
//            photon.direction = refractDirection(photon.direction, normal);
//            photon.direction = sampleHenyeyGreenstein(state, uAnisotropy, photon.direction);
            photon.bounces++;
        } else {
            // null collision
        }
//    }

    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, float(photon.wavelength));
    float t = 1.0 - photon.wavelengthRadiance;
    vec3 color = mix(vec3(1.0), wavelengthToRGB(photon.wavelength), photon.radiance.g);
    oRadiance = vec4(photon.radiance, float(photon.samples));
    oWavelength = vec4(photon.wavelength, photon.wavelengthRadiance, photon.wavelengthTransmittance, 0);
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
//    unprojectRand(state, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);
    unproject(vPosition, uMvpInverseMatrix, from, to);


    photon.direction = normalize(to - from);
    vec2 tbounds = max(intersectCube(from, photon.direction), 0.0);
    photon.position = from + tbounds.x * photon.direction;
    photon.transmittance = vec3(1);
    photon.radiance = vec3(1);
    photon.bounces = 0u;
    photon.samples = 0u;
    photon.wavelength = random_uniform(state) * (720.0 - 320.0) + 320.0;
    photon.wavelengthRadiance = 1.0;
    photon.wavelengthTransmittance = 1.0;
    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, float(photon.wavelength));
    oRadiance = vec4(photon.radiance, float(photon.samples));
    oWavelength = vec4(photon.wavelength, photon.wavelengthRadiance, photon.wavelengthTransmittance, 0);
}
