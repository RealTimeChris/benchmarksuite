#	MIT License
#
#	Copyright (c) 2024 RealTimeChris
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy of this
#	software and associated documentation files (the "Software"), to deal in the Software
#	without restriction, including without limitation the rights to use, copy, modify, merge,
#	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#	persons to whom the Software is furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all copies or
#	substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#	DEALINGS IN THE SOFTWARE.

set(MAIN_SOURCE $<IF:$<BOOL:${BNCH_SWT_DEPLOY}>,main_deploy,main>${BNCH_SWT_MAIN_FILE_EXTENSION})

add_executable(benchmarksuite-main ${MAIN_SOURCE})

if (PRINT_ASSEMBLY)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(STATUS "Assembly printing enabled. Using Post-Build step to generate assembly file.")
        
        if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
            add_custom_command(
                TARGET benchmarksuite-main POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:benchmarksuite-main>"
                COMMAND ${CMAKE_COMMAND} -E echo "Assembly file location: $<TARGET_FILE_DIR:benchmarksuite-main>/benchmarksuite-main.asm"
                COMMAND sh -c "otool -tV '$<TARGET_FILE:benchmarksuite-main>' > '$<TARGET_FILE_DIR:benchmarksuite-main>/benchmarksuite-main.asm'"
                COMMENT "Generating final assembly for benchmarksuite-main"
                VERBATIM
            )
        else()
            add_custom_command(
                TARGET benchmarksuite-main POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:benchmarksuite-main>"
                COMMAND ${CMAKE_COMMAND} -E echo "Assembly file location: $<TARGET_FILE_DIR:benchmarksuite-main>/benchmarksuite-main.asm"
                COMMAND sh -c "objdump -d '$<TARGET_FILE:benchmarksuite-main>' > '$<TARGET_FILE_DIR:benchmarksuite-main>/benchmarksuite-main.asm'"
                COMMENT "Generating final assembly for benchmarksuite-main"
                VERBATIM
            )
        endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(ASM_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/asm_output")
        file(MAKE_DIRECTORY ${ASM_OUTPUT_DIR})

        target_compile_options(benchmarksuite-main PRIVATE 
            "/FAcs"
            "/Fa${ASM_OUTPUT_DIR}/"
        )
    
        message(STATUS "Assembly printing enabled for MSVC. Assembly files will be generated in: ${ASM_OUTPUT_DIR}")

        add_custom_command(
            TARGET benchmarksuite-main POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying MSVC assembly files from ${ASM_OUTPUT_DIR}..."
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASM_OUTPUT_DIR}" "$<TARGET_FILE_DIR:benchmarksuite-main>"
            COMMAND ${CMAKE_COMMAND} -E echo "Assembly files location: $<TARGET_FILE_DIR:benchmarksuite-main>"
            COMMENT "Copying MSVC assembly output"
        )
    endif()
endif()

target_link_libraries(benchmarksuite-main
    PUBLIC
        benchmarksuite::benchmarksuite
)

install(
    FILES
        $<TARGET_FILE:benchmarksuite-main>
    DESTINATION bin
    OPTIONAL
)
