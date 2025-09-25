#define _USE_MATH_DEFINES
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NX 2000
#define NY 2000
#define DX 6.248999f
#define DY 7.397143f
// #define DX 3.952561f
// #define DY 4.680042f
#define DT 0.1f 
#define G 9.81f
#define THRESHOLD 0.001f
#define MAX_VIS_WATER_DEPTH 10.0f
#define DIFFUSION_COEFF 0.1f
#define TOTAL_SIMULATION_DURATION_SECONDS 1000.0f
#define SLOPE_TOLERANCE 1e-7f
#define LOG_INTERVAL 100

#define IDF_PARAMS_A 13837.9124203334f
#define IDF_PARAMS_B 10.0f
#define IDF_PARAMS_C 0.7f
#define STATIC_RAIN_DURATION_MINUTES 60.0f
#define SCS_TOTAL_RAINFALL_MM 100.0f
#define SCS_STORM_DURATION_HOURS 24.0f
#define SCS_PEAK_TIME_RATIO 0.4f
#define ABM_STORM_DURATION_MINUTES 120.0f
#define ABM_BLOCK_DURATION_MINUTES 5.0f

float z[NY][NX];
float eta[NY][NX];
float n_manning[NY][NX];
int valid_cell[NY][NX];
float rain_source[NY][NX];
float overlay_data[NY][NX];
float eta_new[NY][NX];
float p[NY][NX];
float q[NY][NX];

float nodata_value = -9999.0f;
float dem_xllcorner, dem_yllcorner, dem_dx, dem_dy;
bool request_screenshot = false;
bool request_data_dump = false;
bool show_overlay = false;
bool paused = false;

float *abm_hyetograph_m_per_s = NULL;
int abm_hyetograph_num_blocks = 0;

// Variabel OpenGL
GLuint shader_program, vao, vbo;
GLuint tex_eta, tex_z, tex_valid_cell, tex_overlay;
GLint loc_zoom_level, loc_view_offset;
GLint loc_tex_eta, loc_tex_z, loc_tex_valid_cell, loc_tex_overlay;
GLint loc_threshold, loc_max_vis_water_depth, loc_show_overlay;
GLint loc_min_z, loc_max_z;
float zoom_level = 1.0f;
float view_offset_x = 0.0f;
float view_offset_y = 0.0f;
bool is_panning = false;
double last_mouse_x, last_mouse_y;

// Variabel baru untuk manajemen waktu yang lebih robust
double program_start_time = 0.0;
double total_pause_time = 0.0;
double pause_start_time = 0.0;
double simulation_finish_time = 0.0; // Waktu saat simulasi selesai

const char *vertex_shader_src = "#version 330 core\n"
    "in vec2 aPos;in vec2 aTexCoord;out vec2 TexCoord;"
    "uniform float u_zoom_level;uniform vec2 u_view_offset;"
    "void main(){gl_Position=vec4((aPos*u_zoom_level)+u_view_offset,0.0,1.0);TexCoord=aTexCoord;}";

const char *fragment_shader_src = "#version 330 core\n"
    "out vec4 FragColor;"
    "in vec2 TexCoord;"
    "uniform sampler2D u_tex_eta;"
    "uniform sampler2D u_tex_z;"
    "uniform sampler2D u_tex_valid_cell;"
    "uniform sampler2D u_tex_overlay;"
    "uniform bool u_show_overlay;"
    "uniform float u_threshold;"
    "uniform float u_max_vis_water_depth;"
    "uniform float u_min_z;"
    "uniform float u_max_z;"
    "void main(){"
    "   vec4 base_color;"
    "   int valid = int(texture(u_tex_valid_cell, TexCoord).r);"
    "   if (valid == 0) {"
    "       base_color = vec4(0.39, 0.39, 0.39, 1.0);" 
    "   } else {"
    "       float eta_val = texture(u_tex_eta, TexCoord).r;"
    "       float z_val = texture(u_tex_z, TexCoord).r;"
    "       float h = eta_val - z_val;"
    "       if (h < u_threshold) {"
    "           if (u_max_z > u_min_z) {"
    "               float norm_z = (z_val - u_min_z) / (u_max_z - u_min_z);"
    "               norm_z = clamp(norm_z, 0.0, 1.0);"
    "               vec3 color_low = vec3(0.1, 0.6, 0.2);"      
    "               vec3 color_mid_low = vec3(0.9, 1.0, 0.4);"   
    "               vec3 color_mid_high = vec3(1.0, 0.6, 0.2);" 
    "               vec3 color_high = vec3(0.7, 0.2, 0.1);"     
    "               vec3 color_peak = vec3(0.8, 0.8, 0.8);"     
    "               vec3 terrain_color;"
    "               if (norm_z < 0.25) {"
    "                   terrain_color = mix(color_low, color_mid_low, norm_z / 0.25);"
    "               } else if (norm_z < 0.5) {"
    "                   terrain_color = mix(color_mid_low, color_mid_high, (norm_z - 0.25) / 0.25);"
    "               } else if (norm_z < 0.75) {"
    "                   terrain_color = mix(color_mid_high, color_high, (norm_z - 0.5) / 0.25);"
    "               } else {"
    "                   terrain_color = mix(color_high, color_peak, (norm_z - 0.75) / 0.25);"
    "               }"
    "               base_color = vec4(terrain_color, 1.0);"
    "           } else {"
    "               base_color = vec4(0.82, 0.71, 0.55, 1.0);" 
    "           }"
    "       } else {"
    "           float norm = clamp(h / u_max_vis_water_depth, 0.0, 1.0);"
    "           vec3 water_color = mix(vec3(0.68, 0.85, 0.9), vec3(0.0, 0.0, 0.55), norm);"
    "           base_color = vec4(water_color, 1.0);"
    "       }"
    "   }"
    "   FragColor = base_color;"
    "   if (u_show_overlay) {"
    "       float overlay_value = texture(u_tex_overlay, TexCoord).r;"
    "       if (overlay_value > 0.5) {" 
    "           FragColor = vec4(1.0, 0.0, 0.0, 1.0);" 
    "       }"
    "   }"
    "}";

void check_shader_compile(GLuint shader) {
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "Shader compile error: %s\n", infoLog);
    }
}

void init_gl(int width, int height) {
    glewInit();
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader_src, NULL);
    glCompileShader(vs);
    check_shader_compile(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader_src, NULL);
    glCompileShader(fs);
    check_shader_compile(fs);

    shader_program = glCreateProgram();
    glAttachShader(shader_program, vs);
    glAttachShader(shader_program, fs);
    glLinkProgram(shader_program);
    glUseProgram(shader_program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    loc_zoom_level = glGetUniformLocation(shader_program, "u_zoom_level");
    loc_view_offset = glGetUniformLocation(shader_program, "u_view_offset");
    loc_tex_eta = glGetUniformLocation(shader_program, "u_tex_eta");
    loc_tex_z = glGetUniformLocation(shader_program, "u_tex_z");
    loc_tex_valid_cell = glGetUniformLocation(shader_program, "u_tex_valid_cell");
    loc_threshold = glGetUniformLocation(shader_program, "u_threshold");
    loc_max_vis_water_depth = glGetUniformLocation(shader_program, "u_max_vis_water_depth");
    loc_tex_overlay = glGetUniformLocation(shader_program, "u_tex_overlay");
    loc_show_overlay = glGetUniformLocation(shader_program, "u_show_overlay");
    loc_min_z = glGetUniformLocation(shader_program, "u_min_z");
    loc_max_z = glGetUniformLocation(shader_program, "u_max_z");
    
    float vertices[] = {
      -1.0f,  1.0f, 0.0f, 0.0f,   
      -1.0f, -1.0f, 0.0f, 1.0f,   
       1.0f,  1.0f, 1.0f, 0.0f,   
       1.0f, -1.0f, 1.0f, 1.0f  
    };
    
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenTextures(1, &tex_eta);
    glBindTexture(GL_TEXTURE_2D, tex_eta);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT, NULL);

    glGenTextures(1, &tex_z);
    glBindTexture(GL_TEXTURE_2D, tex_z);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT, z);

    float* valid_cell_float = (float*)malloc(NX * NY * sizeof(float));
    if (valid_cell_float) {
        for(int j=0; j<NY; ++j) {
            for(int i=0; i<NX; ++i) {
                valid_cell_float[j*NX + i] = (float)valid_cell[j][i];
            }
        }
        glGenTextures(1, &tex_valid_cell);
        glBindTexture(GL_TEXTURE_2D, tex_valid_cell);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT, valid_cell_float);
        free(valid_cell_float);
    }

    glGenTextures(1, &tex_overlay);
    glBindTexture(GL_TEXTURE_2D, tex_overlay);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT, overlay_data);
}

void render_gl() {
    glBindTexture(GL_TEXTURE_2D, tex_eta);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, NY, GL_RED, GL_FLOAT, eta);

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader_program);
    
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_eta); glUniform1i(loc_tex_eta, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_z); glUniform1i(loc_tex_z, 1);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_valid_cell); glUniform1i(loc_tex_valid_cell, 2);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_overlay); glUniform1i(loc_tex_overlay, 3);
    
    glUniform1f(loc_zoom_level, zoom_level); 
    glUniform2f(loc_view_offset, view_offset_x, view_offset_y);
    glUniform1f(loc_threshold, THRESHOLD); 
    glUniform1f(loc_max_vis_water_depth, MAX_VIS_WATER_DEPTH);
    glUniform1i(loc_show_overlay, show_overlay);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

float sign_host(float x) {
    return (x > 0.0f) - (x < 0.0f);
}

void compute_flux_serial() {
    for (int j = 1; j < NY - 1; j++) {
        for (int i = 1; i < NX - 1; i++) {
            
            if (!valid_cell[j][i]) {
                p[j][i] = 0.0f;
                q[j][i] = 0.0f;
                continue;
            }

            p[j][i] = 0.0f;
            if (valid_cell[j][i+1]) {
                float d_eta = eta[j][i+1] - eta[j][i];

                if (fabsf(d_eta) > SLOPE_TOLERANCE) {
                    float z_face = fmaxf(z[j][i], z[j][i+1]);
                    float eta_up = fmaxf(eta[j][i], eta[j][i+1]);
                    float h_face = eta_up - z_face;

                    if (h_face > THRESHOLD) {
                        float n_avg = 0.5f * (n_manning[j][i] + n_manning[j][i+1]);
                        float flux_adv = 0.0f;
                        if (n_avg > 1e-6f) {
                            float S = d_eta / DX;
                            float v_manning = (1.0f / n_avg) * powf(h_face, 2.0f/3.0f) * sqrtf(fabsf(S));
                            float v_crit = sqrtf(G * h_face);
                            float v_final = fminf(v_manning, v_crit);
                            flux_adv = -v_final * h_face * sign_host(S);
                        }

                        float h_L = fmaxf(0.0f, eta[j][i] - z[j][i]);
                        float h_R = fmaxf(0.0f, eta[j][i+1] - z[j][i+1]);
                        float avg_h_for_diff = 0.5f * (h_L + h_R);
                        float D = DIFFUSION_COEFF * DX * sqrtf(G * avg_h_for_diff);
                        float flux_diff = -D * (h_R - h_L) / DX;
                        
                        p[j][i] = flux_adv + flux_diff;
                    }
                }
            }

            q[j][i] = 0.0f;
            if (valid_cell[j+1][i]) {
                float d_eta = eta[j+1][i] - eta[j][i];

                if (fabsf(d_eta) > SLOPE_TOLERANCE) {
                    float z_face = fmaxf(z[j][i], z[j+1][i]);
                    float eta_up = fmaxf(eta[j][i], eta[j+1][i]);
                    float h_face = eta_up - z_face;

                    if (h_face > THRESHOLD) {
                        float n_avg = 0.5f * (n_manning[j][i] + n_manning[j+1][i]);
                        float flux_adv = 0.0f;
                        if (n_avg > 1e-6f) {
                            float S = d_eta / DY;
                            float v_manning = (1.0f / n_avg) * powf(h_face, 2.0f/3.0f) * sqrtf(fabsf(S));
                            float v_crit = sqrtf(G * h_face);
                            float v_final = fminf(v_manning, v_crit);
                            flux_adv = -v_final * h_face * sign_host(S);
                        }

                        float h_S = fmaxf(0.0f, eta[j][i] - z[j][i]);
                        float h_N = fmaxf(0.0f, eta[j+1][i] - z[j+1][i]);
                        float avg_h_for_diff = 0.5f * (h_S + h_N);
                        float D = DIFFUSION_COEFF * DY * sqrtf(G * avg_h_for_diff);
                        float flux_diff = -D * (h_N - h_S) / DY;
                        
                        q[j][i] = flux_adv + flux_diff;
                    }
                }
            }
        }
    }
}

void update_eta_serial(float dt) {
    for (int j = 1; j < NY - 1; j++) {
        for (int i = 1; i < NX - 1; i++) {

            if (!valid_cell[j][i]) {
                eta_new[j][i] = eta[j][i];
                continue;
            }

            float p_in = p[j][i-1];
            float p_out = p[j][i];
            float q_in = q[j-1][i];
            float q_out = q[j][i];

            float dpdx = (p_out - p_in) / DX;
            float dqdy = (q_out - q_in) / DY;

            float rain_term = rain_source[j][i];

            float temp_eta_new = eta[j][i] - dt * (dpdx + dqdy) + dt * rain_term;

            if (isfinite(temp_eta_new)) {
                eta_new[j][i] = fmaxf(z[j][i], temp_eta_new);
            } else {
                eta_new[j][i] = eta[j][i];
            }
        }
    }
}

void simulate_step_serial(float dt) {
    memcpy(eta_new, eta, sizeof(eta));
    memset(p, 0, sizeof(p));
    memset(q, 0, sizeof(q));
    
    compute_flux_serial();

    update_eta_serial(dt);

    memcpy(eta, eta_new, sizeof(eta));
}

int read_dem(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Gagal membuka file DEM: %s\n", filename);
        return 0;
    }
    int ncols, nrows;
    fscanf(file, "ncols %d\n", &ncols);
    fscanf(file, "nrows %d\n", &nrows);
    fscanf(file, "xllcorner %f\n", &dem_xllcorner);
    fscanf(file, "yllcorner %f\n", &dem_yllcorner);
    fscanf(file, "dx %f\n", &dem_dx);
    fscanf(file, "dy %f\n", &dem_dy);
    fscanf(file, "NODATA_value %f\n", &nodata_value);
    
    if (ncols != NX || nrows != NY) {
        fprintf(stderr, "Error: Dimensi DEM (%dx%d) tidak cocok dengan NXxNY (%dx%d).\n", ncols, nrows, NX, NY);
        fclose(file);
        return 0;
    }
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            float value;
            if (fscanf(file, "%f", &value) != 1 || value == nodata_value) {
                z[j][i] = 0.0f;
                eta[j][i] = 0.0f;
                valid_cell[j][i] = 0;
            } else {
                z[j][i] = value;
                eta[j][i] = value;
                valid_cell[j][i] = 1;
            }
            rain_source[j][i] = 0.0f;
        }
    }
    fclose(file);
    return 1;
}

int read_manning_map(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Gagal membuka file peta Manning: %s\n", filename);
        return 0;
    }
    int ncols, nrows;
    float xllcorner, yllcorner, dx_header, dy_header, manning_nodata_value;
    fscanf(file, "ncols %d\n", &ncols);
    fscanf(file, "nrows %d\n", &nrows);
    fscanf(file, "xllcorner %f\n", &xllcorner);
    fscanf(file, "yllcorner %f\n", &yllcorner);
    fscanf(file, "dx %f\n", &dx_header);
    fscanf(file, "dy %f\n", &dy_header);
    fscanf(file, "NODATA_value %f\n", &manning_nodata_value);
    if (ncols != NX || nrows != NY) {
        fprintf(stderr, "Error: Dimensi peta Manning (%dx%d) tidak cocok dengan NXxNY (%dx%dx).\n", ncols, nrows, NX, NY);
        fclose(file);
        return 0;
    }
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            float value;
            if (fscanf(file, "%f", &value) != 1 || value == manning_nodata_value) {
                n_manning[j][i] = 0.0f;
            } else {
                n_manning[j][i] = value;
            }
            if (n_manning[j][i] < 0.0f) n_manning[j][i] = 0.0f;
        }
    }
    fclose(file);
    printf("Peta Koefisien Manning '%s' berhasil dibaca.\n", filename);
    return 1;
}

int read_overlay_asc(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Peringatan: Gagal membuka file overlay '%s'.\n", filename);
        return 0;
    }
    int ncols, nrows;
    char line[256];
    fscanf(file, "ncols %d\n", &ncols);
    fscanf(file, "nrows %d\n", &nrows);
    if (ncols != NX || nrows != NY) {
        fprintf(stderr, "Peringatan: Dimensi overlay tidak cocok dengan DEM.\n");
        fclose(file);
        return 0;
    }
    for(int i=0; i<5; ++i) fgets(line, sizeof(line), file); 
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            fscanf(file, "%f", &overlay_data[j][i]);
        }
    }
    fclose(file);
    printf("File overlay '%s' berhasil dibaca.\n", filename);
    return 1;
}

void save_screenshot(GLFWwindow* window, const char* filename) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    unsigned char* pixels = (unsigned char*)malloc(width * height * 3);
    if (pixels) {
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        stbi_flip_vertically_on_write(1);
        stbi_write_png(filename, width, height, 3, pixels, width * 3);
        free(pixels);
        printf("Screenshot disimpan ke: %s\n", filename);
    }
}

void save_eta_as_asc(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Gagal membuat file .asc: %s\n", filename);
        return;
    }

    fprintf(file, "ncols %d\n", NX);
    fprintf(file, "nrows %d\n", NY);
    fprintf(file, "xllcorner %f\n", dem_xllcorner);
    fprintf(file, "yllcorner %f\n", dem_yllcorner);
    fprintf(file, "dx %f\n", dem_dx);
    fprintf(file, "dy %f\n", dem_dy);
    fprintf(file, "NODATA_value %f\n", nodata_value);

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            if (valid_cell[j][i]) {
                fprintf(file, "%f ", eta[j][i]);
            } else {
                fprintf(file, "%f ", nodata_value);
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Data Eta disimpan ke: %s\n", filename);
}

float rand_float(unsigned int *seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return ((float)(*seed) / 0x7fffffff);
}
float rain_intensity_random(float time, unsigned int *seed) {
    return 0.05f * rand_float(seed);
}
float rain_intensity_moving(float time) {
    int center_x = (int)(NX / 2 + 100 * sinf(time));
    int center_y = NY / 2;
    float intensity = 0.005f;
    float radius = 100.0f;
    float rain = 0.0f;

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            if (!valid_cell[j][i]) continue;
            float dx_dist = i - center_x;
            float dy_dist = j - center_y;
            float distance = sqrtf(dx_dist * dx_dist + dy_dist * dy_dist);
            if (distance < radius) {
                rain_source[j][i] = intensity * (1.0f - distance / radius);
                rain += rain_source[j][i];
            } else {
                rain_source[j][i] = 0.0f;
            }
        }
    }
    return rain;
}
float rain_intensity_chicago(float time) {
    float peak = 0.01f;
    float duration = 50.0f;
    float t = fmodf(time, duration);
    return peak * expf(-0.1f * t);
}
float calculate_idf_intensity_mm_per_hour(float duration_minutes, float idf_a, float idf_b, float idf_c) {
    if (duration_minutes + idf_b <= 0) return 0.0f;
    return idf_a / powf(duration_minutes + idf_b, idf_c);
}
int compare_floats_desc(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa < fb) - (fa > fb);
}
void precompute_abm_hyetograph(float *hyetograph_array, int num_blocks_to_fill, float idf_a, float idf_b, float idf_c, float storm_duration_minutes, float block_duration_minutes) {
    if (storm_duration_minutes <= 0 || block_duration_minutes <= 0 || hyetograph_array == NULL || num_blocks_to_fill <= 0) return;
    int total_storm_blocks = (int)ceilf(storm_duration_minutes / block_duration_minutes);
    if (total_storm_blocks == 0) return;
    float *incremental_rainfall_mm = (float *)malloc(total_storm_blocks * sizeof(float));
    if (!incremental_rainfall_mm) return;
    float prev_cumulative_rainfall_mm = 0.0f;
    for (int k = 0; k < total_storm_blocks; k++) {
        float current_duration_minutes = (k + 1) * block_duration_minutes;
        float cumulative_intensity_mm_per_hour = calculate_idf_intensity_mm_per_hour(current_duration_minutes, idf_a, idf_b, idf_c);
        float cumulative_rainfall_mm = cumulative_intensity_mm_per_hour * current_duration_minutes / 60.0f;
        incremental_rainfall_mm[k] = cumulative_rainfall_mm - prev_cumulative_rainfall_mm;
        prev_cumulative_rainfall_mm = cumulative_rainfall_mm;
        if (incremental_rainfall_mm[k] < 0.0f) incremental_rainfall_mm[k] = 0.0f;
    }
    qsort(incremental_rainfall_mm, total_storm_blocks, sizeof(float), compare_floats_desc);
    for (int i = 0; i < num_blocks_to_fill; ++i) hyetograph_array[i] = 0.0f;
    int current_hyetograph_idx_left, current_hyetograph_idx_right;
    if (num_blocks_to_fill % 2 != 0) {
        current_hyetograph_idx_left = num_blocks_to_fill / 2;
        current_hyetograph_idx_right = num_blocks_to_fill / 2;
    } else {
        current_hyetograph_idx_left = num_blocks_to_fill / 2 - 1;
        current_hyetograph_idx_right = num_blocks_to_fill / 2;
    }
    int blocks_placed = 0;
    for (int k = 0; k < total_storm_blocks; k++) {
        if (blocks_placed >= num_blocks_to_fill) break;
        float current_block_rainfall_mm = incremental_rainfall_mm[k];
        float current_block_intensity_m_per_s = current_block_rainfall_mm / (block_duration_minutes * 60.0f * 1000.0f);
        if (num_blocks_to_fill % 2 == 0) {
            if (k % 2 == 0) {
                if (current_hyetograph_idx_left >= 0) {
                    hyetograph_array[current_hyetograph_idx_left--] = current_block_intensity_m_per_s;
                    blocks_placed++;
                }
            } else {
                if (current_hyetograph_idx_right < num_blocks_to_fill) {
                    hyetograph_array[current_hyetograph_idx_right++] = current_block_intensity_m_per_s;
                    blocks_placed++;
                }
            }
        } else {
            if (k == 0) {
                hyetograph_array[current_hyetograph_idx_left] = current_block_intensity_m_per_s;
                blocks_placed++;
            } else if (k % 2 == 1) {
                current_hyetograph_idx_right++;
                if (current_hyetograph_idx_right < num_blocks_to_fill) {
                    hyetograph_array[current_hyetograph_idx_right] = current_block_intensity_m_per_s;
                    blocks_placed++;
                }
            } else {
                current_hyetograph_idx_left--;
                if (current_hyetograph_idx_left >= 0) {
                    hyetograph_array[current_hyetograph_idx_left] = current_block_intensity_m_per_s;
                    blocks_placed++;
                }
            }
        }
    }
    free(incremental_rainfall_mm);
}
float get_scs_type2_simplified_intensity_mm_per_hour(float time_in_hours, float total_rainfall_mm, float storm_duration_hours, float peak_time_ratio) {
    if (storm_duration_hours <= 0 || total_rainfall_mm <= 0) return 0.0f;
    float peak_time_hours = storm_duration_hours * peak_time_ratio;
    float current_intensity_mm_per_hour = 0.0f;
    if (time_in_hours >= 0 && time_in_hours <= storm_duration_hours) {
        if (time_in_hours <= peak_time_hours) {
            current_intensity_mm_per_hour = (2.0f * total_rainfall_mm / storm_duration_hours) * (time_in_hours / peak_time_hours);
        } else {
            current_intensity_mm_per_hour = (2.0f * total_rainfall_mm / storm_duration_hours) * ((storm_duration_hours - time_in_hours) / (storm_duration_hours - peak_time_hours));
        }
    }
    if (current_intensity_mm_per_hour < 0.0f) current_intensity_mm_per_hour = 0.0f;
    return current_intensity_mm_per_hour;
}
float compute_rainfall(int step, float time, int pattern) { 
    float total_rain = 0.0f; 
    unsigned int seed = (unsigned int)(time * 1000); 
    float current_rain_intensity_m_per_s = 0.0f; 
    switch (pattern) { 
        case 0: 
            for (int j = 0; j < NY; j++) {
                unsigned int local_seed = seed + j; 
                for (int i = 0; i < NX; i++) {
                    if (!valid_cell[j][i]) continue;
                    rain_source[j][i] = rain_intensity_random(time, &local_seed) / 3.6e6f;
                    total_rain += rain_source[j][i];
                } 
            } 
            break; 
        case 1: 
            rain_intensity_moving(time); 
            for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) if (valid_cell[j][i]) total_rain += rain_source[j][i]; 
            break; 
        case 2: case 3: case 4: case 5: 
            if (pattern == 2) { current_rain_intensity_m_per_s = rain_intensity_chicago(time) / 3.6e6f; } 
            else if (pattern == 3) { current_rain_intensity_m_per_s = calculate_idf_intensity_mm_per_hour(STATIC_RAIN_DURATION_MINUTES, IDF_PARAMS_A, IDF_PARAMS_B, IDF_PARAMS_C) / 3.6e6f; } 
            else if (pattern == 4) { current_rain_intensity_m_per_s = get_scs_type2_simplified_intensity_mm_per_hour(time / 3600.0f, SCS_TOTAL_RAINFALL_MM, SCS_STORM_DURATION_HOURS, SCS_PEAK_TIME_RATIO) / 3.6e6f; } 
            else if (pattern == 5) { int idx = (int)((time / 60.0f) / ABM_BLOCK_DURATION_MINUTES); current_rain_intensity_m_per_s = (idx < abm_hyetograph_num_blocks) ? abm_hyetograph_m_per_s[idx] : 0.0f; } 
            for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) if (valid_cell[j][i]) { rain_source[j][i] = current_rain_intensity_m_per_s; total_rain += rain_source[j][i]; } 
            break; 
        default: 
            for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) rain_source[j][i] = 0.0f; 
            total_rain = 0.0f; 
            break; 
    } 
    return total_rain * DX * DY; 
}

void write_log(FILE *logfile, int step, float time, float rain_volume, float *out_max_h, float *out_min_h) {
    float eta_volume = 0.0f, max_h = -INFINITY, min_h = INFINITY;
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            if (!valid_cell[j][i]) continue;
            float h = eta[j][i] - z[j][i];
            if (h > 0.0f) {
                eta_volume += h * DX * DY;
            }
            if (h > max_h) max_h = h;
            if (h < min_h) min_h = h;
        }
    }
    fprintf(logfile, "%d,%.2f,%f,%f,%f,%f\n", step, time, rain_volume, eta_volume, max_h, min_h);
    if (out_max_h) *out_max_h = max_h;
    if (out_min_h) *out_min_h = min_h;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    float zoom_factor = 1.1f;
    if (yoffset > 0) {
        zoom_level *= zoom_factor;
    } else {
        zoom_level /= zoom_factor;
    }
    if (zoom_level < 0.1f) zoom_level = 0.1f;
    if (zoom_level > 100.0f) zoom_level = 100.0f;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            is_panning = true;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        } else if (action == GLFW_RELEASE) {
            is_panning = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    if (is_panning) {
        double dx = xpos - last_mouse_x;
        double dy = ypos - last_mouse_y;
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        view_offset_x += (float)(dx / width) * 2.0f / zoom_level;
        view_offset_y -= (float)(dy / height) * 2.0f / zoom_level;

        float max_offset_x = (1.0f - 1.0f / zoom_level);
        float max_offset_y = (1.0f - 1.0f / zoom_level);
        if (view_offset_x > max_offset_x) view_offset_x = max_offset_x;
        if (view_offset_x < -max_offset_x) view_offset_x = -max_offset_x;
        if (view_offset_y > max_offset_y) view_offset_y = max_offset_y;
        if (view_offset_y < -max_offset_y) view_offset_y = -max_offset_y;

        last_mouse_x = xpos;
        last_mouse_y = ypos;
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_S) {
            request_screenshot = true;
        } else if (key == GLFW_KEY_D) {
            request_data_dump = true;
        } else if (key == GLFW_KEY_O) { 
            show_overlay = !show_overlay;
            printf("Tampilan Overlay: %s\n", show_overlay ? "Aktif" : "Nonaktif");
        } else if (key == GLFW_KEY_SPACE) {
            paused = !paused;
            printf("Simulasi %s\n", paused ? "DIJEDA" : "DILANJUTKAN");

            // Logika baru untuk manajemen waktu
            if (paused) {
                // Saat dijeda, catat waktu mulai jeda
                pause_start_time = glfwGetTime();
            } else {
                // Saat dilanjutkan, tambahkan durasi jeda ke total waktu jeda
                if (pause_start_time > 0.0) {
                    total_pause_time += glfwGetTime() - pause_start_time;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    float current_dt = DT;
    float current_sim_time = 0.0f;
    int total_steps_taken = 0;

    printf("Program Simulasi Banjir - Versi Serial (CPU).\n");
    printf("Versi: Interaktif dengan visualisasi modern.\n");

    if (argc < 5) {
        printf("Usage: %s <dem.asc> <manning.asc> <rain_pattern> <overlay.asc>\n", argv[0]);
        return 1;
    }
    
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }

    if (!read_dem(argv[1])) { glfwTerminate(); return 1; }
    if (!read_manning_map(argv[2])) { glfwTerminate(); return 1; }
    int rain_pattern = atoi(argv[3]);
    read_overlay_asc(argv[4]); 

    float min_z = INFINITY, max_z = -INFINITY;
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            if (valid_cell[j][i]) {
                if (z[j][i] < min_z) min_z = z[j][i];
                if (z[j][i] > max_z) max_z = z[j][i];
            }
        }
    }
    printf("Elevasi Terrain: Min = %.2f m, Max = %.2f m\n", min_z, max_z);

    if (rain_pattern == 5) {
        abm_hyetograph_num_blocks = (int)ceilf(ABM_STORM_DURATION_MINUTES / ABM_BLOCK_DURATION_MINUTES);
        if (abm_hyetograph_num_blocks > 0) {
            abm_hyetograph_m_per_s = (float *)malloc(abm_hyetograph_num_blocks * sizeof(float));
            if (abm_hyetograph_m_per_s) {
                precompute_abm_hyetograph(abm_hyetograph_m_per_s, abm_hyetograph_num_blocks, IDF_PARAMS_A, IDF_PARAMS_B, IDF_PARAMS_C, ABM_STORM_DURATION_MINUTES, ABM_BLOCK_DURATION_MINUTES);
            }
        }
    }

    int window_width = 1024, window_height = 1024;
    GLFWwindow *window = glfwCreateWindow(window_width, window_height, "Simulasi Banjir Serial", NULL, NULL);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    init_gl(window_width, window_height);
    
    glUseProgram(shader_program);
    glUniform1f(loc_min_z, min_z);
    glUniform1f(loc_max_z, max_z);

    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetKeyCallback(window, key_callback);

    FILE *logfile = fopen("log_simulasi_serial.csv", "w");
    fprintf(logfile, "step,time,rain_volume,eta_volume,max_h,min_h\n");
    float current_max_h = 0.0f, current_min_h = 0.0f;
    
    bool simulation_finished = false;
    // Catat waktu mulai efektif dari loop simulasi
    program_start_time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        if (!paused && !simulation_finished) {
            if (current_sim_time >= TOTAL_SIMULATION_DURATION_SECONDS) {
                simulation_finished = true;
                simulation_finish_time = glfwGetTime(); // Catat waktu simulasi selesai
                printf("\n--- SIMULASI SELESAI ---\n");
            } else {
                current_dt = DT; 

                if (total_steps_taken > 0 && total_steps_taken % LOG_INTERVAL == 0) {
                    if (!isfinite(current_max_h)) {
                        fprintf(stderr, "\n---! KETIDAKSTABILAN NUMERIK TERDETEKSI !---\n");
                        break;
                    }
                }

                float rain = compute_rainfall(total_steps_taken, current_sim_time, rain_pattern);
                simulate_step_serial(current_dt);

                current_sim_time += current_dt;
                if (total_steps_taken % LOG_INTERVAL == 0) {
                    write_log(logfile, total_steps_taken, current_sim_time, rain, &current_max_h, &current_min_h);
                }
                total_steps_taken++;
            }
        }

        render_gl();
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        if (request_screenshot) {
            char filename[256];
            snprintf(filename, sizeof(filename), "screenshot_serial_step_%d.png", total_steps_taken);
            save_screenshot(window, filename);
            request_screenshot = false; 
        }
        if (request_data_dump) {
            char filename[256];
            snprintf(filename, sizeof(filename), "eta_serial_step_%d.asc", total_steps_taken);
            save_eta_as_asc(filename);
            request_data_dump = false; 
        }

        char title_buffer[256];
        const char* status_text = paused ? "DIJEDA" : (simulation_finished ? "SELESAI" : "BERJALAN");
        snprintf(title_buffer, sizeof(title_buffer), "Simulasi Serial | Status: %s | Time: %.2f s | Max h: %.4f m", 
                 status_text, current_sim_time, current_max_h);
        glfwSetWindowTitle(window, title_buffer);
    }

    double program_end_time = glfwGetTime();

    // Jika program ditutup saat sedang dijeda, tambahkan durasi jeda terakhir
    if (paused) {
        total_pause_time += program_end_time - pause_start_time;
    }

    // Tentukan waktu akhir efektif untuk perhitungan
    double effective_end_time;
    if (simulation_finish_time > 0.0) { // Jika simulasi selesai normal
        effective_end_time = simulation_finish_time;
    } else { // Jika simulasi diinterupsi (jendela ditutup)
        effective_end_time = program_end_time;
    }
    
    double total_runtime = effective_end_time - program_start_time;
    double execution_time = total_runtime - total_pause_time;

    if (abm_hyetograph_m_per_s != NULL) free(abm_hyetograph_m_per_s);
    fclose(logfile);
    glfwDestroyWindow(window);
    glfwTerminate();

    printf("\n--- Laporan Performa Simulasi ---\n");
    printf("Waktu Eksekusi (Simulasi Aktif) : %.4f detik\n", execution_time);
    printf("Total Langkah Simulasi          : %d langkah\n", total_steps_taken);
    if (total_steps_taken > 0 && execution_time > 0.0001) { // Hindari pembagian dengan nol
        printf("Waktu Rata-rata/Langkah         : %.6f detik/langkah\n", execution_time / total_steps_taken);
        printf("Langkah/Detik (LPS) Rata-rata   : %.2f langkah/detik\n", total_steps_taken / execution_time);
    }
    printf("--------------------------------------------------------\n");
    
    return 0;
}

