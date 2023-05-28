// #part /glsl/shaders/TransferFunction/vertex

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

// #part /glsl/shaders/TransferFunction/fragment

#version 300 es
precision mediump float;

uniform vec2 uPosition;
uniform vec2 uSize;
uniform vec4 uColor;

in vec2 vPosition;

out vec4 oColor;

vec3 HSVtoRGB(in vec3 HSV)
{
    float H   = HSV.x;
    float R   = abs(H * 6.0 - 3.0) - 1.0;
    float G   = 2.0 - abs(H * 6.0 - 2.0);
    float B   = 2.0 - abs(H * 6.0 - 4.0);
    vec3  RGB = clamp( vec3(R,G,B), 0.0, 1.0 );
    return ((RGB - 1.0) * HSV.y + 1.0) * HSV.z;
}

void main() {
//    float r = length((uPosition - vPosition) / uSize);
//    float cutoff = uPosition.x;
//    if (vPosition.x < cutoff)
//        oColor = vec4(0, 0, 0, 0);
//    else
//        oColor = vec4(HSVtoRGB(vec3((vPosition.x-cutoff) * (1.0 + cutoff), 1, 1)), 1);

//    void main() {
        //    float r = length((uPosition - vPosition) / uSize);
        //    float cutoff = uPosition.x;
        //    if (vPosition.x < cutoff)
        //        oColor = vec4(0, 0, 0, 0);
        //    else
        //        oColor = vec4(HSVtoRGB(vec3((vPosition.x-cutoff) * (1.0 + cutoff), 1, 1)), 1);
        //
        //    if (vPosition.x > uPosition.x - uSize.x && vPosition.x < uPosition.x + uSize.x) {
        //        vec4 spectrum = vec4(HSVtoRGB(vec3(vPosition.y, 1, 1)), 1);
        //        float r = length((uPosition.x - vPosition.x));
        //        spectrum = spectrum * (1.0 - r / uSize.x);
        //        oColor = spectrum;
        //    }
        //    else {
        //        oColor = vec4(0, 0, 0, 0);
        //
        //    }
//        vec4 spectrum = vec4(HSVtoRGB(vec3(vPosition.y, 1, 1)), 1);
//        float r = length((uPosition - vPosition) / uSize);
//        //    if (vPosition.x > 0.5 && vPosition.y > 0.5) {
//        //        oColor = vec4(HSVtoRGB(vec3(vPosition.x, 1, 1)), 1);
//        //    } else {
//        oColor = uColor * exp(-r * r);

        //    }
//    }

    float r = length((uPosition - vPosition) / uSize);
    oColor = uColor * exp(-r * r);
//    oColor = vec4(HSVtoRGB(vec3(vPosition.y, 1, 1)), 1) * exp(-r * r);

//    if (vPosition.y > 0.95) {
//        oColor = vec4(0, 1.0, 0, 1);
//    } else {
//        oColor = uColor * exp(-r * r);
//    }
//    else {
//    if (vPosition.y > 0.85) {
//        oColor = uColor;
//    } else {
//        oColor = vec4(0, 0, 0, 0);
//    }
}
