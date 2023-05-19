const { vec2 } = require('./gl-matrix-min.js');

// Initialize the image to 0 radiance.
const width = 256;
const height = 128;
const image = [];
for (let i = 0; i < width; i++) {
    image[i] = [];
    for (let j = 0; j < height; j++) {
        image[i][j] = 0;
    }
}

// Define two circles for our volume.
function insideLeftCircle(position) {
    return vec2.squaredDistance(position, [1 / 4, 1 / 2]) < (1 / 4) * (1 / 4);
}

function insideRightCircle(position) {
    return vec2.squaredDistance(position, [3 / 4, 1 / 2]) < (1 / 4) * (1 / 4);
}

// Scattering everywhere, except inside the circles.
function scattering(position) {
    if (insideLeftCircle(position) || insideRightCircle(position)) {
        return 0;
    } else {
        return 20;
    }
}

// An absorbing circle on the left half of the image.
// Also, absorption has to be nonzero for the emission to work.
function absorption(position) {
    if (insideLeftCircle(position)) {
        return 20;
    } else if (insideRightCircle(position)) {
        return 100;
    } else {
        return 0;
    }
}

// An emitting circle on the right half of the image.
function emission(position) {
    if (insideRightCircle(position)) {
        return 100;
    } else {
        return 0;
    }
}

// The constant mu, which has to be greater than than
// the absorption and scattering coefficients combined.
const density = 100;

// This has been converted to iteration from recursion, because Node
// evidently does not optimize tail calls.
function estimateRadiance(position, direction) {
    while (true) {
        // First, sample the free-flight distance of the photon.
        const distance = -Math.log(Math.random()) / density;
        vec2.scaleAndAdd(position, position, direction, -distance);

        // If the photon ends up outside our image, we can set the radiance of
        // the background to 0. Can also be different for "environment mapping".
        const [x, y] = position;
        if (x < 0 || x > 1 || y < 0 || y > 1) {
            return 0;
        }

        // Second, set event probabilities based on the absorption
        // and scattering coefficients at the new position.
        // These probabilities can actually be almost anything, but if
        // we choose them this way, we do not have to keep track of a
        // weight that would result from probabilistic evaluation.
        const probabilityAbsorption = absorption(position) / density;
        const probabilityScattering = scattering(position) / density;

        // Finally, based on the above probabilities, randomly
        // choose an event and simulate it.
        const event = Math.random();
        if (event < probabilityAbsorption) {
            // In the case of absorption/emission, just return emission.
            return emission(position);
        } else if (event < probabilityAbsorption + probabilityScattering) {
            // In the case of scattering, choose a random direction and continue.
            vec2.random(direction);
        } else {
            // In the case of a null collision, nothing happens,
            // and the photon continues from the new position.
        }
    }
}

// Estimate radiance for every pixel with the set number of samples.
const samples = 64;
for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
        for (let sample = 0; sample < samples; sample++) {
            // Place the photon randomly inside the pixel.
            const position = [(i + Math.random()) / width, (j + Math.random()) / height];

            // Choose a random direction of the photon.
            const direction = vec2.random(vec2.create());

            // Calculate the average of the radiance estimates.
            image[i][j] += estimateRadiance(position, direction) / samples;
        }
    }
}

// We write the image to standard output in ASCII PGM format
// (P2 of the NetPBM suite), because it is trivial to write.
console.log('P2');
console.log(`${width} ${height}`);
console.log(255);
for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
        // Apply exposure and gamma correction before output.
        const exposure = 0.01;
        const gamma = 2.2;
        const brightness = Math.pow(image[i][j] * exposure, 1 / gamma) * 255;
        console.log(Math.round(Math.min(Math.max(brightness, 0), 255)));
    }
}