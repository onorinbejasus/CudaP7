#include "common/util/config_reader.hpp"
#include "common/log.hpp"
#include <cstdlib>

namespace aed {
namespace common {
namespace util {


ConfigNode::ConfigNode(const TiXmlElement* node)
{
    this->node = node;
}

ConfigNode::~ConfigNode() { }


const ConfigNode* ConfigNode::get_node(const char* node_name, int id) const
{
    if(node == NULL)
        return NULL;

    const TiXmlElement* child = node->FirstChildElement();
    while(child != NULL) {
        if (child->ValueStr() == node_name) {
            int child_id;
            int rv = child->QueryIntAttribute("id", &child_id);
            
            if (rv == TIXML_SUCCESS && child_id == id || rv == TIXML_NO_ATTRIBUTE) {
                return new ConfigNode(child);
            }
        }
        child = child->NextSiblingElement();
    }

    return NULL;
}

const ConfigNode* ConfigNode::get_node(const char* node_name, const char* name) const
{
    if(node == NULL)
        return NULL;

    const TiXmlElement* child = node->FirstChildElement();
    while(child != NULL) {
        if (child->ValueStr() == node_name) {
            string attrString;
            int rv = child->QueryStringAttribute("name",&attrString);
            
            if (rv == TIXML_SUCCESS && attrString == name) {
                return new ConfigNode(child);
            }
        }
        child = child->NextSiblingElement();
    }

    return NULL;

}

// In a list of nodes without ids, gets the nth node
const ConfigNode* ConfigNode::get_node_num(const char* node_name, int num) const
{
    int count = 0;
    
    if(node == NULL)
        return NULL;
    
    const TiXmlElement* child = node->FirstChildElement();
    while(child != NULL) {
        if (child->ValueStr() == node_name) {
            if (count == num) {
                return new ConfigNode(child);
            }
            
            count++;
        }
        child = child->NextSiblingElement();
    }
    
    return NULL;
}

int ConfigNode::get_node_count(const char* node_name) const
{
    int count = 0;
    
    if(node == NULL)
        return count;
    
    const TiXmlElement* child = node->FirstChildElement();
    while(child != NULL) {
        if (child->ValueStr() == node_name) {
            count++;
        }
        child = child->NextSiblingElement();
    }
    
    return count;
}

string ConfigNode::get_value(const char* setting_name) const
{
    const TiXmlElement* child = node->FirstChildElement();

    while (child != NULL) {
        if(child->ValueStr() == setting_name) {
            return child->GetText();
        }
        child = child->NextSiblingElement();
    }

    char msg[100];
    snprintf(msg, 100, "Could not find \"%s\" node in config.", setting_name);
    throw ParseErrorException(msg);
}



bool ConfigReader::load_config_file(const char* filename)
{
    bool rv = doc.LoadFile( filename );
    if ( !rv ) {
        LOG_VAR_MSG( MDL_BACK_END, SVR_ERROR, "Parse error while loading configuration file %s at %d,%d:", 
                filename, doc.ErrorRow(), doc.ErrorCol());
        LOG_MSG( MDL_BACK_END, SVR_ERROR, doc.ErrorDesc() );
    }
    return rv;
}


const ConfigNode* ConfigReader::get_root_node() const
{
    return new ConfigNode(doc.RootElement());
}


} // util
} // common
} // namespace aed


