#ifndef AED_CONFIG_READER_HPP
#define AED_CONFIG_READER_HPP

#include "tinyxml/tinyxml.h"

using std::string;

namespace aed {
namespace common {
namespace util {

class ConfigNode
{
public:
    ConfigNode(const TiXmlElement* node);
    ~ConfigNode();

    const ConfigNode* get_node(const char* node_name, int id = 0) const;
    const ConfigNode* get_node(const char* node_name, const char* settings_name) const;
    
    const ConfigNode* get_node_num(const char* node_name, int num) const;
    int get_node_count(const char* node_name) const;
    
    string get_value(const char* setting_name) const;

    const TiXmlElement* node;
};


class ConfigReader
{
public:
    ConfigReader() { }
    ~ConfigReader() { }

    bool load_config_file(const char* filename);
    const ConfigNode* get_root_node() const;

private:
    TiXmlDocument doc;
};



} // util
} // common
} // namespace aed

#endif // AED_CONFIG_READER_HPP

