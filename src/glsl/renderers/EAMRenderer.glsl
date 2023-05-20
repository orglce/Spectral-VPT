// #part /glsl/shaders/renderers/EAM/generate/vertex

#version 300 es

uniform mat4 uMvpInverseMatrix;

out vec3 vRayFrom;
out vec3 vRayTo;

// #link /glsl/mixins/unproject.glsl
@unproject

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

void main() {
    vec2 position = vertices[gl_VertexID];
    unproject(position, uMvpInverseMatrix, vRayFrom, vRayTo);
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/EAM/generate/fragment

#version 300 es
precision mediump float;
precision mediump sampler2D;
precision mediump sampler3D;

uniform sampler3D uVolume;
uniform sampler2D uTransferFunction;
uniform float uStepSize;
uniform float uOffset;
uniform float uExtinction;

in vec3 vRayFrom;
in vec3 vRayTo;

out vec4 oColor;

float x[10] = float[](
0.021767,
0.289098,
0.134957,
0.076227,
0.559137,
1.003337,
0.512631,
0.069413,
0.004594,
0.000269);

float y[10] = float[](
0.000608,
0.022317,
0.139052,
0.640193,
0.971281,
0.660025,
0.210725,
0.025454,
0.001659,
0.000009);

float z[10] = float[](
0.10375,
1.46738,
0.97180,
0.12021,
0.00734,
0.00089,
0.00005,
0.00000,
0.00000,
0.00000);

// #link /glsl/mixins/intersectCube.glsl
@intersectCube

vec4[10] sampleVolumeColor(vec3 position) {
    vec2 volumeSample = texture(uVolume, position).rg;

    vec4[10] specter;

    for (int i = 0; i < 10; i++) {
        specter[i] = texture(uTransferFunction, vec2(volumeSample.r, float(i)/10.0));
    }

    return specter;
}

void main() {
    vec3 rayDirection = vRayTo - vRayFrom;
    vec2 tbounds = max(intersectCube(vRayFrom, rayDirection), 0.0);
    if (tbounds.x >= tbounds.y) {
        oColor = vec4(0, 0, 0, 1);
    } else {
        vec3 from = mix(vRayFrom, vRayTo, tbounds.x);
        vec3 to = mix(vRayFrom, vRayTo, tbounds.y);
        float rayStepLength = distance(from, to) * uStepSize;

        float t = uStepSize * uOffset;
        vec4 accumulator = vec4(0);

        while (t < 1.0 && accumulator.a < 0.99) {
            vec3 position = mix(from, to, t);

            vec4[10] colorSample = sampleVolumeColor(position);

            float x_val = 0.0;
            float y_val = 0.0;
            float z_val = 0.0;

            for (int i = 0; i < 10; i++) {
                x_val += x[i] * colorSample[i].a;
                y_val += y[i] * colorSample[i].a;
                z_val += z[i] * colorSample[i].a;
            }

            float R = 3.2404542 * x_val - 1.5371385 * y_val - 0.4985314 * z_val;
            float G = -0.9692660 * x_val + 1.8760108 * y_val + 0.0415560 * z_val;
            float B = 0.0556434 * x_val - 0.2040259 * y_val + 1.0572252 * z_val;

            if (R < 0.0) R = 0.0;
            if (G < 0.0) G = 0.0;
            if (B < 0.0) B = 0.0;
            if (R > 1.0) R = 1.0;
            if (G > 1.0) G = 1.0;
            if (B > 1.0) B = 1.0;

            vec4 max_color = vec4(R, G, B, 1.0);

//            vec4 max_color = vec4(x_val, x_val, x_val, 1.0);
//            vec4 max_color = colorSample[9];

//            vec4 max_color = vec4(0);
//            float max_alpha = 0.0;
//            for (int i = 0; i < 10; i++) {
//                if (colorSample[i].a > max_alpha) {
//                    max_alpha = colorSample[i].a;
//                    max_color = colorSample[i];
//                }
//            }
            max_color = colorSample[9];
            max_color.a *= rayStepLength * uExtinction;
            max_color.rgb *= max_color.a;
            accumulator += (1.0 - accumulator.a) * max_color;
            t += uStepSize;
        }

        if (accumulator.a > 1.0) {
            accumulator.rgb /= accumulator.a;
        }

        oColor = vec4(accumulator.rgb, 1);
    }
}

// #part /glsl/shaders/renderers/EAM/integrate/vertex

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

// #part /glsl/shaders/renderers/EAM/integrate/fragment

#version 300 es
precision mediump float;
precision mediump sampler2D;
precision mediump sampler3D;

uniform sampler2D uAccumulator;
uniform sampler2D uFrame;
uniform float uMix;

in vec2 vPosition;

out vec4 oColor;

void main() {
    vec4 accumulator = texture(uAccumulator, vPosition);
    vec4 frame = texture(uFrame, vPosition);
    oColor = mix(accumulator, frame, uMix);
}

// #part /glsl/shaders/renderers/EAM/render/vertex

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

// #part /glsl/shaders/renderers/EAM/render/fragment

#version 300 es
precision mediump float;
precision mediump sampler2D;
precision mediump sampler3D;

uniform sampler2D uAccumulator;
uniform sampler3D uVolume;
uniform sampler2D uTransferFunction;

in vec2 vPosition;

out vec4 oColor;

vec4[10] sampleVolumeColor(vec3 position) {
    vec2 volumeSample = texture(uVolume, position).rg;

    vec4[10] specter;

    for (int i = 0; i < 10; i++) {
        specter[i] = texture(uTransferFunction, vec2(volumeSample.r, float(i)/10.0));
    }

    return specter;
}

void main() {
    oColor = texture(uAccumulator, vPosition);
}

// #part /glsl/shaders/renderers/EAM/reset/vertex

#version 300 es

const vec2 vertices[] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

void main() {
    vec2 position = vertices[gl_VertexID];
    gl_Position = vec4(position, 0, 1);
}

// #part /glsl/shaders/renderers/EAM/reset/fragment

#version 300 es
precision mediump float;

out vec4 oColor;

void main() {
    oColor = vec4(0, 0, 0, 1);
}
