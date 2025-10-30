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


#ifdef _OPENMP
    #include <omp.h>
    #define GET_TIME() omp_get_wtime()
#else
    #define omp_get_max_threads() 1
    #define GET_TIME() ((double)clock() / CLOCKS_PER_SEC)
#endif

// #define NX 3162
// #define NY 3162
// #define DX 3.952561f
// #define DY 4.680042f
#define NX 1000
#define NY 1000
#define DY 12.497999f
#define DX 14.794286f
#define DT 0.1f
#define G 9.81f
#define THRESHOLD 0.001f
#define MAX_VIS_WATER_DEPTH 10.0f

#define DIFFUSION_COEFF 0.1f

//#define MIN_DT 1e-8f
#define TOTAL_SIMULATION_DURATION_SECONDS 1000.0f
#define SLOPE_TOLERANCE 1e-7f
#define LOG_INTERVAL 100

#define PRD_INITIAL_INTENSITY_MM_HR 150.0f 
#define PRD_DECAY_FACTOR 0.25f            

#define IDF_PARAMS_A 13837.9124203334f
#define IDF_PARAMS_B 10.0f
#define IDF_PARAMS_C 0.7f
#define ABM_STORM_DURATION_MINUTES 120.0f
#define ABM_BLOCK_DURATION_MINUTES 5.0f

#define SCS_TOTAL_RAINFALL_MM 400.0f 
#define SCS_PEAK_TIME_RATIO 0.4f

float z[NY][NX];
float eta[NY][NX];
float eta_new[NY][NX];
float p[NY][NX];
float q[NY][NX];
float rain_source[NY][NX];
int valid_cell[NY][NX];
float nodata_value = -9999.0f;
float n_manning[NY][NX];
float dem_xllcorner, dem_yllcorner, dem_dx, dem_dy;
bool request_screenshot = false;
bool request_data_dump = false;
float overlay_data[NY][NX];
bool show_overlay = false;
bool paused = false;

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

float *abm_hyetograph_m_per_s = NULL;
int abm_hyetograph_num_blocks = 0;

// Shader OpenGL
const char *vertex_shader_src = "#version 330 core\nin vec2 aPos;in vec2 aTexCoord;out vec2 TexCoord;uniform float u_zoom_level;uniform vec2 u_view_offset;void main(){gl_Position=vec4((aPos*u_zoom_level)+u_view_offset,0.0,1.0);TexCoord=aTexCoord;}";
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

double total_paused_time = 0.0;
double pause_start_time = 0.0;

void check_shader_compile(GLuint shader) {
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "Shader compile error: %s\n", infoLog);
    }
}

int read_overlay_asc(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Peringatan: Gagal membuka file overlay '%s'. Overlay tidak akan ditampilkan.\n", filename);
        return 0;
    }
    int ncols, nrows;
    char line[256];
    fscanf(file, "ncols %d\n", &ncols);
    fscanf(file, "nrows %d\n", &nrows);
    if (ncols != NX || nrows != NY) {
        fprintf(stderr, "Peringatan: Dimensi overlay (%dx%d) tidak cocok dengan DEM (%dx%d). Overlay tidak akan ditampilkan.\n", ncols, nrows, NX, NY);
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
      -1.0f,  1.0f,       0.0f,  0.0f,   
      -1.0f, -1.0f,       0.0f,  1.0f,   
       1.0f,  1.0f,       1.0f,  0.0f,   
       1.0f, -1.0f,       1.0f,  1.0f  
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
    
    glUniform1f(loc_zoom_level, zoom_level); glUniform2f(loc_view_offset, view_offset_x, view_offset_y);
    glUniform1f(loc_threshold, THRESHOLD); glUniform1f(loc_max_vis_water_depth, MAX_VIS_WATER_DEPTH);
    glUniform1i(loc_show_overlay, show_overlay);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
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

float rain_intensity_chicago(float time_in_seconds) {
    float time_in_hours = time_in_seconds / 3600.0f;
    return PRD_INITIAL_INTENSITY_MM_HR * expf(-PRD_DECAY_FACTOR * time_in_hours);
}


float calculate_idf_intensity_mm_per_hour(float duration_minutes, float idf_a, float idf_b, float idf_c) {
    if (duration_minutes + idf_b <= 0) {
        return 0.0f;
    }
    return idf_a / powf(duration_minutes + idf_b, idf_c);
}

int compare_floats_desc(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa < fb) - (fa > fb);
}

void precompute_abm_hyetograph(float *hyetograph_array, int num_blocks_to_fill, float idf_a, float idf_b, float idf_c, float storm_duration_minutes, float block_duration_minutes) {
    if (storm_duration_minutes <= 0 || block_duration_minutes <= 0 || hyetograph_array == NULL || num_blocks_to_fill <= 0) {
        return;
    }

    int total_storm_blocks = (int)ceilf(storm_duration_minutes / block_duration_minutes);
    if (total_storm_blocks == 0) {
        return;
    }

    float *incremental_rainfall_mm = (float *)malloc(total_storm_blocks * sizeof(float));
    if (!incremental_rainfall_mm) {
        return;
    }

    float prev_cumulative_rainfall_mm = 0.0f;
    for (int k = 0; k < total_storm_blocks; k++) {
        float current_duration_minutes = (k + 1) * block_duration_minutes;
        float cumulative_intensity_mm_per_hour = calculate_idf_intensity_mm_per_hour(current_duration_minutes, idf_a, idf_b, idf_c);
        float cumulative_rainfall_mm = cumulative_intensity_mm_per_hour * current_duration_minutes / 60.0f;
        incremental_rainfall_mm[k] = cumulative_rainfall_mm - prev_cumulative_rainfall_mm;
        prev_cumulative_rainfall_mm = cumulative_rainfall_mm;
        if (incremental_rainfall_mm[k] < 0.0f) {
            incremental_rainfall_mm[k] = 0.0f;
        }
    }

    qsort(incremental_rainfall_mm, total_storm_blocks, sizeof(float), compare_floats_desc);

    for (int i = 0; i < num_blocks_to_fill; ++i) {
        hyetograph_array[i] = 0.0f;
    }

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
        if (blocks_placed >= num_blocks_to_fill) {
            break;
        }
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
    if (storm_duration_hours <= 0 || total_rainfall_mm <= 0) {
        return 0.0f;
    }
    float peak_time_hours = storm_duration_hours * peak_time_ratio;
    float current_intensity_mm_per_hour = 0.0f;
    if (time_in_hours >= 0 && time_in_hours <= storm_duration_hours) {
        if (time_in_hours <= peak_time_hours) {
            current_intensity_mm_per_hour = (2.0f * total_rainfall_mm / storm_duration_hours) * (time_in_hours / peak_time_hours);
        } else {
            if (storm_duration_hours - peak_time_hours > 1e-6) {
                current_intensity_mm_per_hour = (2.0f * total_rainfall_mm / storm_duration_hours) * ((storm_duration_hours - time_in_hours) / (storm_duration_hours - peak_time_hours));
            }
        }
    }
    if (current_intensity_mm_per_hour < 0.0f) {
        current_intensity_mm_per_hour = 0.0f;
    }
    return current_intensity_mm_per_hour;
}


// --- FUNGSI SIMULASI SERIAL (PENGGANTI KERNEL CUDA) ---

// Fungsi untuk menghitung hujan di CPU
void compute_rainfall_cpu(int pattern, float time, float uniform_intensity) {
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            if (!valid_cell[j][i]) {
                rain_source[j][i] = 0.0f;
                continue;
            }

            switch (pattern) {
                case 0: 
                    rain_source[j][i] = (0.05f * ((float)rand() / RAND_MAX)) / 3.6e6f;
                    break;
                case 1: { 
                    float center_x = (float)(NX / 2 + 100 * sinf(time));
                    float center_y = (float)(NY / 2);
                    float intensity = 0.005f;
                    float radius = 100.0f;
                    float dx_dist = (float)i - center_x;
                    float dy_dist = (float)j - center_y;
                    float distance = sqrtf(dx_dist * dx_dist + dy_dist * dy_dist);
                    if (distance < radius) {
                        rain_source[j][i] = intensity * (1.0f - distance / radius);
                    } else {
                        rain_source[j][i] = 0.0f;
                    }
                    break;
                }
                case 2: // Peaked
                case 3: // IDF random 
                case 4: // SCS 
                case 5: // ABM 
                    rain_source[j][i] = uniform_intensity;
                    break;
                default:
                    rain_source[j][i] = 0.0f;
                    break;
            }
        }
    }
}


float sign_host(float x) {
    return (x > 0.0f) - (x < 0.0f);
}

// fluks CPU
void compute_flux_cpu() {
    for (int j = 1; j < NY - 1; j++) {
        for (int i = 1; i < NX - 1; i++) {

            if (!valid_cell[j][i]) {
                p[j][i] = 0.0f;
                q[j][i] = 0.0f;
                continue;
            }

            // fluks p
            p[j][i] = 0.0f;
            if (valid_cell[j][i+1]) {
                float d_eta = eta[j][i+1] - eta[j][i];

                if (fabsf(d_eta) > SLOPE_TOLERANCE) {
                    float z_face = fmaxf(z[j][i], z[j][i+1]);
                    float eta_up = fmaxf(eta[j][i], eta[j][i+1]);
                    float h_face = eta_up - z_face;

                    if (h_face > THRESHOLD) {
                        float flux_adv = 0.0f;
                        float n_avg = 0.5f * (n_manning[j][i] + n_manning[j][i+1]);
                        if (n_avg > 1e-6f) {
                            float S = d_eta / DX;
                            float v_manning = (1.0f / n_avg) * powf(h_face, 0.6666667f) * sqrtf(fabsf(S));
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

            // fluks q
            q[j][i] = 0.0f;
            if (valid_cell[j+1][i]) {
                float d_eta = eta[j+1][i] - eta[j][i];

                if (fabsf(d_eta) > SLOPE_TOLERANCE) {
                    float z_face = fmaxf(z[j][i], z[j+1][i]);
                    float eta_up = fmaxf(eta[j][i], eta[j+1][i]);
                    float h_face = eta_up - z_face;

                    if (h_face > THRESHOLD) {
                        float flux_adv = 0.0f;
                        float n_avg = 0.5f * (n_manning[j][i] + n_manning[j+1][i]);
                        if (n_avg > 1e-6f) {
                            float S = d_eta / DY;
                            float v_manning = (1.0f / n_avg) * powf(h_face, 0.6666667f) * sqrtf(fabsf(S));
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

// update eta
void update_eta_cpu(float dt) {
    for (int j = 1; j < NY - 1; j++) {
        for (int i = 1; i < NX - 1; i++) {
            if (!valid_cell[j][i]) {
                eta_new[j][i] = eta[j][i];
                continue;
            }

            float p_in = p[j][i - 1];
            float p_out = p[j][i];
            float q_in = q[j - 1][i];
            float q_out = q[j][i];

            float dpdx = (p_out - p_in) / DX;
            float dqdy = (q_out - q_in) / DY;

            float rain_term = rain_source[j][i];

            float temp_eta_new = eta[j][i] - dt * (dpdx + dqdy) + dt * rain_term;

            if (isfinite(temp_eta_new)) {
                if (temp_eta_new < z[j][i]) {
                    eta_new[j][i] = z[j][i];
                } else {
                    eta_new[j][i] = temp_eta_new;
                }
            } else {
                eta_new[j][i] = eta[j][i];
            }
        }
    }
}

void simulate_step_cpu(float dt) {
    compute_flux_cpu();
    update_eta_cpu(dt);
    memcpy(eta, eta_new, sizeof(eta));
}


void write_log(FILE *logfile, int step, float time, float intensity_mm_hr, float cumulative_rainfall_mm, float *out_max_h, float *out_min_h) {
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
    fprintf(logfile, "%d,%.2f,%.4f,%.4f,%f,%.6f,%.6f\n", step, time, intensity_mm_hr, cumulative_rainfall_mm, eta_volume, max_h, min_h);
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
        if (key == GLFW_KEY_S) { request_screenshot = true; } 
        else if (key == GLFW_KEY_D) { request_data_dump = true; } 
        else if (key == GLFW_KEY_O) { 
            show_overlay = !show_overlay;
            printf("Tampilan Overlay: %s\n", show_overlay ? "Aktif" : "Nonaktif");
        } else if (key == GLFW_KEY_SPACE) {
            paused = !paused;
            if (paused) {
                pause_start_time = GET_TIME();
                printf("Simulasi DIJEDA\n");
            } else {
                if (pause_start_time > 0.0) {
                    total_paused_time += (GET_TIME() - pause_start_time);
                }
                printf("Simulasi DILANJUTKAN\n");
            }
        }
    }
}

int main(int argc, char *argv[]) {
    float current_dt = DT;
    float current_sim_time = 0.0f;
    int total_steps_taken = 0;

    double total_computation_time = 0.0;
    double total_rendering_time = 0.0;
    
    float cumulative_rainfall_mm = 0.0f;


    printf("Program dikompilasi untuk eksekusi SERIAL di CPU.\n");
    printf("Versi: Interaktif dengan Kalkulasi di CPU\n");

    if (argc < 5) {
        printf("Usage: %s <dem.asc> <manning.asc> <rain_pattern> <overlay.asc>\n", argv[0]);
        return 1;
    }
    if (!read_dem(argv[1])) { return 1; }
    if (!read_manning_map(argv[2])) { return 1; }
    int rain_pattern = atoi(argv[3]);
    read_overlay_asc(argv[4]); 

    // random generator untuk pola hujan dummy
    srand(time(NULL));

    float min_z = INFINITY;
    float max_z = -INFINITY;
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

    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    int window_width = 1024, window_height = 1024;
    GLFWwindow *window = glfwCreateWindow(window_width, window_height, "Simulasi Banjir CPU", NULL, NULL);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    init_gl(window_width, window_height);
    glUseProgram(shader_program);
    glUniform1f(loc_min_z, min_z);
    glUniform1f(loc_max_z, max_z);

    // Alokasi memori sudah ditangani oleh deklarasi array global,
    // tidak perlu alokasi dinamis atau transfer ke GPU.

    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetKeyCallback(window, key_callback);

    FILE *logfile = fopen("log_simulasi_serial.csv", "w");
    fprintf(logfile, "step,time,intensity(mm/hr),cumulative_rainfall(mm),eta_volume,max_h,min_h\n");
    float current_max_h = 0.0f, current_min_h = 0.0f;
    
    bool simulation_finished = false;
    double simulation_end_time = 0.0;
    bool is_end_time_recorded = false;
    double start_time = GET_TIME();
    
    const float scs_dynamic_duration_hours = TOTAL_SIMULATION_DURATION_SECONDS / 3600.0f;

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        
        if (!paused && !simulation_finished) {
            
            if (current_sim_time >= TOTAL_SIMULATION_DURATION_SECONDS) {
                simulation_finished = true;
            
                if (!is_end_time_recorded) {
                    simulation_end_time = GET_TIME();
                    is_end_time_recorded = true;
                }

                printf("\n--- SIMULASI SELESAI ---\n");
                printf("Durasi total %.2f detik telah tercapai.\n", TOTAL_SIMULATION_DURATION_SECONDS);
                printf("Jendela akan tetap terbuka. Tekan ESC atau tutup jendela untuk keluar.\n");
            } else {
                
                double computation_start = GET_TIME();

                float uniform_rain_intensity_m_per_s = 0.0f;
                if (rain_pattern == 2) { 
                    uniform_rain_intensity_m_per_s = rain_intensity_chicago(current_sim_time) / 3.6e6f; 
                } 
                else if (rain_pattern == 4) { 
                    uniform_rain_intensity_m_per_s = get_scs_type2_simplified_intensity_mm_per_hour(current_sim_time / 3600.0f, SCS_TOTAL_RAINFALL_MM, scs_dynamic_duration_hours, SCS_PEAK_TIME_RATIO) / 3.6e6f; 
                } 
                else if (rain_pattern == 5) { 
                    int idx = (int)((current_sim_time / 60.0f) / ABM_BLOCK_DURATION_MINUTES); 
                    uniform_rain_intensity_m_per_s = (idx < abm_hyetograph_num_blocks) ? abm_hyetograph_m_per_s[idx] : 0.0f; 
                } 
                
                float current_intensity_mm_hr = uniform_rain_intensity_m_per_s * 3600.0f * 1000.0f;
                float rain_depth_this_step_mm = (uniform_rain_intensity_m_per_s * DT) * 1000.0f;
                cumulative_rainfall_mm += rain_depth_this_step_mm;

                compute_rainfall_cpu(rain_pattern, current_sim_time, uniform_rain_intensity_m_per_s);
                
                simulate_step_cpu(DT);

                double computation_end = GET_TIME();
                total_computation_time += (computation_end - computation_start);
                
                current_sim_time += DT;
                total_steps_taken++;

                double render_start = GET_TIME();
                if (total_steps_taken % LOG_INTERVAL == 0) {
                    write_log(logfile, total_steps_taken, current_sim_time, current_intensity_mm_hr, cumulative_rainfall_mm, &current_max_h, &current_min_h);
                }
                render_gl();
                double render_end = GET_TIME();
                total_rendering_time += (render_end - render_start);
            }
        }

        double event_start = GET_TIME();
        glfwSwapBuffers(window);
        glfwPollEvents();
        double event_end = GET_TIME();
        if (!paused && !simulation_finished) {
             total_rendering_time += (event_end - event_start);
        }
        
        if (request_screenshot) {
            char filename[256];
            snprintf(filename, sizeof(filename), "screenshot_step_%d_time_%.2f_maxh_%.4f.png", 
                     total_steps_taken, current_sim_time, current_max_h);
            save_screenshot(window, filename);
            request_screenshot = false; 
        }
        if (request_data_dump) {
            char filename[256];
            snprintf(filename, sizeof(filename), "eta_data_step_%d_time_%.2f.asc", 
                     total_steps_taken, current_sim_time);
            save_eta_as_asc(filename);
            request_data_dump = false; 
        }

        char title_buffer[256];
        const char* status_text = paused ? "DIJEDA" : (simulation_finished ? "SELESAI" : "BERJALAN");
        snprintf(title_buffer, sizeof(title_buffer), "Simulasi Banjir (CPU) | Status: %s | Time: %.2f s | Max h: %.4f m | S=SS, D=Data, O=Overlay, Spasi=Pause", 
                 status_text, current_sim_time, current_max_h);
        glfwSetWindowTitle(window, title_buffer);
    }

    if (!is_end_time_recorded) {
        simulation_end_time = GET_TIME();
    }
    double wall_clock_time_used = (simulation_end_time - start_time) - total_paused_time;

    if (abm_hyetograph_m_per_s != NULL) free(abm_hyetograph_m_per_s);
    fclose(logfile);

    glfwDestroyWindow(window);
    glfwTerminate();

    printf("\n--- Laporan Performa Simulasi Serial ---\n");
    printf("Total Waktu Eksekusi (Wall Clock) : %.4f detik\n", wall_clock_time_used);
    printf("Total Langkah Simulasi            : %d langkah\n\n", total_steps_taken);
    
    printf("--- Rincian Waktu Komputasi ---\n");
    printf("Waktu Komputasi (Simulasi & Hujan): %.4f detik (%.2f%%)\n", total_computation_time, (total_computation_time / wall_clock_time_used) * 100.0);
    printf("Waktu Rendering & GUI             : %.4f detik (%.2f%%)\n", total_rendering_time, (total_rendering_time / wall_clock_time_used) * 100.0);
    
    double unmeasured_time = wall_clock_time_used - (total_computation_time + total_rendering_time);
    if (unmeasured_time > 0.0) {
        printf("Waktu Lain-lain (Tidak Terukur)   : %.4f detik (%.2f%%)\n", unmeasured_time, (unmeasured_time / wall_clock_time_used) * 100.0);
    }
    printf("--------------------------------------------------------\n");

    return 0;
}
