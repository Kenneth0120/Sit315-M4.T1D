#include <mpi.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <map>
#include <algorithm>
#include <cctype>  // For tolower

// Constants for MPI communication
#define TERMINATE_TAG -1  // Signal for termination
#define ACK_TAG 1  // Acknowledgement tag for communication success
#define REDUCE_TAG 2  // Tag for reduction operation during result aggregation

// Colors for output
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[94m"
#define COLOR_GREEN "\033[32m"
#define COLOR_RESET "\033[0m"

// Maximum retries for fault tolerance
#define MAX_RETRIES 3
#define TIMEOUT 5 // Timeout for MPI communications in seconds

std::mutex cout_mutex;  // Mutex for synchronizing output to avoid race conditions
std::map<std::string, int> keyword_sentiments;  // A map to store keyword sentiments

// Function to load keyword sentiments from a file
// Each line of the file contains a word and its corresponding sentiment value
void load_sentiments_from_file(const std::string& file_name) {
    std::ifstream file(file_name);
    std::string word;
    int sentiment;
    while (file >> word >> sentiment) {
        keyword_sentiments[word] = sentiment;  // Insert word and sentiment value into map
    }
}

// Function to check if a word is a negation (e.g., "not", "no")
// Used to modify the sentiment calculation logic
bool is_negation(const std::string& word) {
    return (word == "not" || word == "never" || word == "no");
}

// Helper function to clean and normalize words
// It converts words to lowercase and removes punctuation marks
std::string clean_word(const std::string& word) {
    std::string cleaned_word = word;
    std::transform(cleaned_word.begin(), cleaned_word.end(), cleaned_word.begin(), ::tolower);
    cleaned_word.erase(std::remove_if(cleaned_word.begin(), cleaned_word.end(), ::ispunct), cleaned_word.end());
    return cleaned_word;
}

// Helper function to send a map from one process to another
// It sends both the map's size and the key-value pairs
void send_map(const std::map<std::string, int>& keyword_count, int dest) {
    int map_size = keyword_count.size();  // Number of entries in the map
    MPI_Send(&map_size, 1, MPI_INT, dest, REDUCE_TAG, MPI_COMM_WORLD);  // Send map size

    for (const auto& [key, value] : keyword_count) {
        int key_length = key.length();  // Length of the key (word)
        MPI_Send(&key_length, 1, MPI_INT, dest, REDUCE_TAG, MPI_COMM_WORLD);
        MPI_Send(key.c_str(), key_length, MPI_CHAR, dest, REDUCE_TAG, MPI_COMM_WORLD);  // Send the key
        MPI_Send(&value, 1, MPI_INT, dest, REDUCE_TAG, MPI_COMM_WORLD);  // Send the value (count)
    }
}

// Helper function to receive a map from another process
// The map is reconstructed by receiving the key-value pairs
void receive_map(std::map<std::string, int>& keyword_count, int source) {
    int map_size;
    MPI_Recv(&map_size, 1, MPI_INT, source, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive map size

    for (int i = 0; i < map_size; ++i) {
        int key_length;
        MPI_Recv(&key_length, 1, MPI_INT, source, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<char> key_data(key_length);  // Prepare buffer to receive the key
        MPI_Recv(key_data.data(), key_length, MPI_CHAR, source, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string key(key_data.begin(), key_data.end());  // Reconstruct the key

        int value;
        MPI_Recv(&value, 1, MPI_INT, source, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        keyword_count[key] = value;  // Insert key-value pair into the map
    }
}

// Function to read a file and split it into chunks for distribution to consumers
// Each chunk will be processed in parallel
std::vector<std::string> split_file_into_chunks(const std::string& file_name, int chunk_size) {
    std::ifstream file(file_name);  // Open the file
    std::vector<std::string> chunks;
    std::string chunk;
    std::string line;
    int current_size = 0;

    // Read the file line by line and accumulate the content into chunks
    while (std::getline(file, line)) {
        current_size += line.size();
        chunk += line + "\n";
        if (current_size >= chunk_size) {
            chunks.push_back(chunk);  // Push completed chunk
            chunk.clear();
            current_size = 0;
        }
    }
    if (!chunk.empty()) {
        chunks.push_back(chunk);  // Push any remaining content as a chunk
    }

    return chunks;  // Return the list of chunks
}

// Function to process a chunk of text
// The chunk is processed to count lines, words, characters, and sentiment scores
void process_file_chunk(const std::string& chunk, long long int& lines, long long int& words, long long int& chars,
    int& sentiment_score, std::map<std::string, int>& keyword_count,
    std::map<std::string, int>& track_keywords) {

    std::istringstream stream(chunk);  // Convert chunk into a stream for line-by-line processing
    std::string line;
    lines = 0;
    words = 0;
    chars = 0;
    sentiment_score = 0;

    while (std::getline(stream, line)) {
        lines++;
        chars += line.size();
        std::istringstream word_stream(line);  // Split line into words
        std::string word;
        bool negation_detected = false;  // Track negations

        while (word_stream >> word) {
            words++;
            word = clean_word(word);  // Clean and normalize the word

            // Handle negations for sentiment adjustment
            if (is_negation(word)) {
                negation_detected = true;
            }
            else if (keyword_sentiments.find(word) != keyword_sentiments.end()) {
                int sentiment = keyword_sentiments[word];
                sentiment_score += (negation_detected ? -sentiment : sentiment);  // Apply negation if detected
                keyword_count[word]++;  // Track word occurrences
                negation_detected = false;  // Reset negation after sentiment word
            }

            // Track specific keywords (good, bad, excellent, etc.)
            if (track_keywords.find(word) != track_keywords.end()) {
                track_keywords[word]++;
            }
        }
    }
}

// Producer function that distributes file chunks to consumers and handles fault tolerance
void producer(int rank, int world_size, const std::string& file_name, int chunk_size) {
    // Initialize counters for total lines, words, characters, and sentiment scores
    long long int total_lines = 0, total_words = 0, total_chars = 0;
    int total_sentiment_score = 0;
    std::map<std::string, int> total_track_keywords = {
        {"good", 0}, {"bad", 0}, {"excellent", 0}, {"terrible", 0}, {"okay", 0}
    };

    // Split the input file into chunks
    std::vector<std::string> chunks = split_file_into_chunks(file_name, chunk_size);

    // Distribute chunks to consumers
    for (int i = 0; i < chunks.size(); ++i) {
        int retries = 0;
        int consumer_rank = (i % (world_size - 1)) + 1;  // Assign consumer rank in a round-robin fashion
        bool success = false;

        while (retries < MAX_RETRIES && !success) {
            int chunk_size = chunks[i].size();
            MPI_Send(&chunk_size, 1, MPI_INT, consumer_rank, 0, MPI_COMM_WORLD);  // Send chunk size to consumer
            MPI_Send(const_cast<char*>(chunks[i].c_str()), chunk_size, MPI_CHAR, consumer_rank, 0, MPI_COMM_WORLD);  // Send chunk data

            // Non-blocking receive with a timeout for acknowledgment
            int ack;
            MPI_Request request;
            MPI_Irecv(&ack, 1, MPI_INT, consumer_rank, ACK_TAG, MPI_COMM_WORLD, &request);

            double start_time = MPI_Wtime();
            while (MPI_Wtime() - start_time < TIMEOUT) {
                int flag;
                MPI_Test(&request, &flag, MPI_STATUS_IGNORE);  // Check if acknowledgment is received
                if (flag) {
                    success = true;
                    break;
                }
            }

            // Retry mechanism if acknowledgment is not received
            if (success) {
                std::lock_guard<std::mutex> cout_lock(cout_mutex);
                std::cout << COLOR_YELLOW << "[Producer] Distributed chunk " << i + 1 << "/" << chunks.size() << COLOR_RESET << std::endl;
            }
            else {
                retries++;
                std::cout << COLOR_RED << "[Producer] Consumer " << consumer_rank << " failed to process chunk " << i + 1
                    << ". Retrying... (" << retries << "/" << MAX_RETRIES << ")" << COLOR_RESET << std::endl;
            }

            // Reassign chunk to another consumer if retries are exhausted
            if (retries == MAX_RETRIES) {
                std::cout << COLOR_RED << "[Producer] Consumer " << consumer_rank << " failed after " << retries
                    << " retries. Reassigning chunk to next consumer." << COLOR_RESET << std::endl;
                consumer_rank = ((i + 1) % (world_size - 1)) + 1;  // Reassign to another consumer
                retries = 0;  // Reset retries for new consumer
            }
        }
    }

    // Send termination signal to all consumers after all chunks are processed
    int terminate_signal = TERMINATE_TAG;
    for (int j = 1; j < world_size; ++j) {
        MPI_Send(&terminate_signal, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
    }

    // Collect results from all consumers (reduction phase)
    for (int j = 1; j < world_size; ++j) {
        long long int lines, words, chars;
        int sentiment;
        MPI_Recv(&lines, 1, MPI_LONG_LONG, j, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&words, 1, MPI_LONG_LONG, j, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&chars, 1, MPI_LONG_LONG, j, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sentiment, 1, MPI_INT, j, REDUCE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        total_lines += lines;
        total_words += words;
        total_chars += chars;
        total_sentiment_score += sentiment;

        // Receive tracked keywords from consumers
        std::map<std::string, int> track_keywords;
        receive_map(track_keywords, j);

        // Update global tracked keyword counts
        for (const auto& [word, count] : track_keywords) {
            total_track_keywords[word] += count;
        }
    }

    // Print the final aggregated results
    std::cout << COLOR_RED << "[Producer] Final processed results - Total Lines: " << total_lines
        << ", Total Words: " << total_words << ", Total Characters: " << total_chars
        << ", Sentiment Score: " << total_sentiment_score << COLOR_RESET << std::endl;

    // Display occurrences of specific keywords
    std::cout << COLOR_GREEN << "Occurrences of 'good': " << total_track_keywords["good"] << "\n"
        << "Occurrences of 'bad': " << total_track_keywords["bad"] << "\n"
        << "Occurrences of 'excellent': " << total_track_keywords["excellent"] << "\n"
        << "Occurrences of 'terrible': " << total_track_keywords["terrible"] << "\n"
        << "Occurrences of 'okay': " << total_track_keywords["okay"] << COLOR_RESET << std::endl;
}

// Consumer function that processes chunks of text and performs sentiment analysis
void consumer(int rank) {
    long long int total_lines = 0, total_words = 0, total_chars = 0;
    int total_sentiment_score = 0;
    std::map<std::string, int> keyword_count;  // Dummy map for function signature
    std::map<std::string, int> track_keywords = {
        {"good", 0}, {"bad", 0}, {"excellent", 0}, {"terrible", 0}, {"okay", 0}
    };

    while (true) {
        int chunk_size;
        MPI_Recv(&chunk_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (chunk_size == TERMINATE_TAG) {
            std::cout << "[Consumer " << rank << "] Terminating..." << std::endl;

            // Send the final results back to the producer
            MPI_Send(&total_lines, 1, MPI_LONG_LONG, 0, REDUCE_TAG, MPI_COMM_WORLD);
            MPI_Send(&total_words, 1, MPI_LONG_LONG, 0, REDUCE_TAG, MPI_COMM_WORLD);
            MPI_Send(&total_chars, 1, MPI_LONG_LONG, 0, REDUCE_TAG, MPI_COMM_WORLD);
            MPI_Send(&total_sentiment_score, 1, MPI_INT, 0, REDUCE_TAG, MPI_COMM_WORLD);
            send_map(track_keywords, 0);  // Send tracked keywords to producer
            break;
        }

        // Receive the chunk data
        std::vector<char> chunk_data(chunk_size);
        MPI_Recv(chunk_data.data(), chunk_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string chunk(chunk_data.begin(), chunk_data.end());

        long long int lines = 0, words = 0, chars = 0;
        int sentiment = 0;
        process_file_chunk(chunk, lines, words, chars, sentiment, keyword_count, track_keywords);  // Process chunk

        // Accumulate results for the final reduction
        total_lines += lines;
        total_words += words;
        total_chars += chars;
        total_sentiment_score += sentiment;

        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, ACK_TAG, MPI_COMM_WORLD);  // Send acknowledgment to producer
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);  // Initialize the MPI environment

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // Get the number of processes in the MPI world

    load_sentiments_from_file("sentiment_dictionary.txt");  // Load the sentiment dictionary

    const std::string file_name = "large_file.txt";
    int chunk_size = 1024;  // Size of each chunk to distribute to consumers

    if (rank == 0) {
        producer(rank, world_size, file_name, chunk_size);  // Producer logic
    }
    else {
        consumer(rank);  // Consumer logic
    }

    MPI_Finalize();  // Finalize the MPI environment
    return 0;
}
