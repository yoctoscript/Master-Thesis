#ifndef LOGGER_HPP
#define LOGGER_HPP
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#define LOGGER_LEVEL spdlog::level::debug

class Logger{
    public:
        Logger(std::string name){
            console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_color(spdlog::level::trace, "white");
            console_sink->set_color(spdlog::level::debug, "cyan");
            console_sink->set_color(spdlog::level::info, "green");
            console_sink->set_color(spdlog::level::warn, "yellow");
            console_sink->set_color(spdlog::level::err, "red");
            console_sink->set_color(spdlog::level::critical, "bold red");
            logger = spdlog::stdout_color_mt(name);
            logger->set_level(LOGGER_LEVEL);
            logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
        }
        std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink;
        std::shared_ptr<spdlog::logger> logger;

        /// Defines a function called trace which takes in any number of arguments and forwards them to the trace method of a logger object.
        template<typename... Args>
        void trace(Args&&... args){
            logger->trace(std::forward<Args>(args)...);
        }

        /// Defines a function called debug which takes in any number of arguments and forwards them to the debug method of a logger object.
        template<typename... Args>
        void debug(Args&&... args){
            logger->debug(std::forward<Args>(args)...);
        }

        /// Defines a function called info which takes in any number of arguments and forwards them to the info method of a logger object.
        template<typename... Args>
        void info(Args&&... args){
            logger->info(std::forward<Args>(args)...);
        }

        /// Defines a function called warn which takes in any number of arguments and forwards them to the warn method of a logger object.
        template<typename... Args>
        void warn(Args&&... args){
            logger->warn(std::forward<Args>(args)...);
        }

        /// Defines a function called error which takes in any number of arguments and forwards them to the error method of a logger object.
        template<typename... Args>
        void error(Args&&... args){
            logger->error(std::forward<Args>(args)...);
        }

        /// Defines a function called critical which takes in any number of arguments and forwards them to the critical method of a logger object.
        template<typename... Args>
        void critical(Args&&... args){
            logger->critical(std::forward<Args>(args)...);
        }
};
#endif